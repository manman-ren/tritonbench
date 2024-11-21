"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import os
import sys

import numpy as np
import torch

import triton
import triton.language as tl

# check if we have the TMA version in Triton PR #4498 (https://github.com/triton-lang/triton/pull/4498).
HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)
WITH_COMPPIPE = os.getenv("ENABLE_COMPPIPE")
PEEL_LAST = os.getenv("PEEL_LAST_ITER")

if HAS_TMA_DESC:
    print(
        "TMA benchmarks will be running with experimental grid constant TMA descriptor.",
        file=sys.stderr,
    )
else:
    print(
        "TMA benchmarks will be running without grid constant TMA descriptor.",
        file=sys.stderr,
    )


class TmaAutoTuneHelper:
    # duck typing wrapper to implement the same interface as TmaDescKernelParam in Triton PR #4498
    class KernelParamWrapper:
        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        self.fill_1d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_1d_tma_descriptor
        )
        self.fill_2d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor
        )
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    # Call this method outside of the lambda function for grid size
    def init_tma_descriptor(self, name):
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8
            )
        else:
            self.cuda_descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8
            )

    # Call this method inside the lambda function for grid size
    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(
        self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size
    ):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name):
        if HAS_TMA_DESC:
            assert self.descriptors[name] is not None
            return self.KernelParamWrapper(self.descriptors[name])
        else:
            assert self.cuda_descriptors[name] is not None
            return self.cuda_descriptors[name]


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    desc_k,
    desc_v,
    Q,
    qvk_offset,
    stride_kn,
    stride_vn,
    stride_vk,  #
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    if not ENABLE_TMA:
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N):  # , loop_schedule=LOOP_SCHEDULE):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if ENABLE_TMA:
            k = tl._experimental_descriptor_load(  # load in row major
                desc_k,
                [start_n.to(tl.int32) + (qvk_offset // stride_kn).to(tl.int32), 0],
                [BLOCK_N, HEAD_DIM],
                Q.dtype.element_ty,
            )
        else:
            k = tl.load(K_block_ptr)
        if ENABLE_TMA:
            k = tl.trans(k)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        if ENABLE_TMA:
            if fp8_v:
                v = tl._experimental_descriptor_load(  # load in row major
                    desc_v,
                    [(qvk_offset // stride_vn).to(tl.int32), start_n.to(tl.int32)],
                    [HEAD_DIM, BLOCK_N],
                    Q.dtype.element_ty,
                )
            else:
                v = tl._experimental_descriptor_load(  # load in row major
                    desc_v,
                    [(qvk_offset // stride_vk + start_n).to(tl.int32), 0],
                    [BLOCK_N, HEAD_DIM],
                    Q.dtype.element_ty,
                )
        else:
            v = tl.load(V_block_ptr)
        if fp8_v:
            if ENABLE_TMA:
                v = tl.trans(v)
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        if not ENABLE_TMA:
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_inner_ws(
    acc,
    l_i,
    m_i,
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    desc_k,
    desc_v,
    Q,
    qvk_offset,
    stride_kn,
    stride_vn,
    stride_vk,  #
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    if not ENABLE_TMA:
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N):  # , loop_schedule=LOOP_SCHEDULE):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        with tl.async_task([0]):
            if ENABLE_TMA:
                k = tl._experimental_descriptor_load(  # load in row major
                    desc_k,
                    [start_n.to(tl.int32) + (qvk_offset // stride_kn).to(tl.int32), 0],
                    [BLOCK_N, HEAD_DIM],
                    Q.dtype.element_ty,
                )
            else:
                k = tl.load(K_block_ptr)
        with tl.async_task([1, 2]):
            if ENABLE_TMA:
                k = tl.trans(k)
            qk = tl.dot(q, k)
            if STAGE == 2:
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                qk -= m_ij[:, None]
            else:
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            # -- update output accumulator --
            acc = acc * alpha[:, None]
            # update acc
        with tl.async_task([0]):
            if ENABLE_TMA:
                if fp8_v:
                    v = tl._experimental_descriptor_load(  # load in row major
                        desc_v,
                        [(qvk_offset // stride_vn).to(tl.int32), start_n.to(tl.int32)],
                        [HEAD_DIM, BLOCK_N],
                        Q.dtype.element_ty,
                    )
                else:
                    v = tl._experimental_descriptor_load(  # load in row major
                        desc_v,
                        [(qvk_offset // stride_vk + start_n).to(tl.int32), 0],
                        [BLOCK_N, HEAD_DIM],
                        Q.dtype.element_ty,
                    )
            else:
                v = tl.load(V_block_ptr)
        with tl.async_task([1, 2]):
            if fp8_v:
                if ENABLE_TMA:
                    v = tl.trans(v)
                p = p.to(tl.float8e5)
            else:
                p = p.to(tl.bfloat16)
            acc = tl.dot(p, v, acc)
            # update m_i and l_i
            m_i = m_ij
        if not ENABLE_TMA:
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Uncommenting
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
has_warp_spec = hasattr(tl, "async_task")
schedList = ["default", "FA_firstDot", "FA_secondDot"] if WITH_COMPPIPE else ["default"]
# TODO: incorrect result with PEEL_LAST + FA_firstDot + WarpSpec + TMA
schedList = ["FA_secondDot"] if PEEL_LAST else schedList
# no WS, no TMA, with CompPipe
configsOpt = [
    (
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "ENABLE_TMA": enable_tma,
                "LOOP_SCHEDULE": sched,
            },
            num_stages=4 if sched == "FA_firstDot" or sched == "FA_secondDot" else 3,
            num_warps=w,
            num_buffers_warp_spec=0,
            num_consumer_groups=0,
        )
        if has_warp_spec
        else triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "ENABLE_TMA": enable_tma,
                "LOOP_SCHEDULE": sched,
            },
            num_stages=4 if sched == "FA_firstDot" or sched == "FA_secondDot" else 3,
            num_warps=w,
        )
    )
    for BM in [128]
    for BN in [128]
    for sched in schedList
    for enable_tma in [False]
    for w in [8]
]
# no WS, with TMA and CompPipe
configsTma = [
    (
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "ENABLE_TMA": enable_tma,
                "LOOP_SCHEDULE": sched,
            },
            num_stages=4 if sched == "FA_firstDot" or sched == "FA_secondDot" else 3,
            num_warps=w,
            num_buffers_warp_spec=0,
            num_consumer_groups=0,
        )
        if has_warp_spec
        else triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "ENABLE_TMA": enable_tma,
                "LOOP_SCHEDULE": sched,
            },
            num_stages=4 if sched == "FA_firstDot" or sched == "FA_secondDot" else 3,
            num_warps=w,
        )
    )
    for BM in [128]
    for BN in [128]
    for sched in schedList
    for enable_tma in [True]
    for w in [8]
]
# no TMA, with WS and CompPipe
configsWS = [
    (
        triton.Config(
            {"BLOCK_M": BM, "BLOCK_N": BN, "ENABLE_TMA": False, "LOOP_SCHEDULE": sched},
            num_stages=2 if sched == "FA_firstDot" or sched == "FA_secondDot" else 0,
            num_warps=w,
            num_buffers_warp_spec=buf,
            num_consumer_groups=grp,
            reg_dec_producer=dec,
            reg_inc_consumer=inc,
        )
        if has_warp_spec
        else triton.Config(
            {"BLOCK_M": BM, "BLOCK_N": BN, "ENABLE_TMA": False, "LOOP_SCHEDULE": sched},
            num_stages=2 if sched == "FA_firstDot" or sched == "FA_secondDot" else 0,
            num_warps=w,
        )
    )
    for BM in [128]
    for BN in [128]
    for sched in schedList
    for enable_ws in [True]
    for w in [4]
    for buf in [2]
    for grp in [2]
    for dec, inc in [(24, 240), (40, 232)]  # 32,240 hangs, 24, 240 works 40, 232 works
]
# BLOCK_M: 128, BLOCK_N: 128, ENABLE_TMA: False, LOOP_SCHEDULE: default, num_warps: 8, num_ctas: 1, num_stages: 3
configsOrig = [
    (
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "ENABLE_TMA": False,
                "LOOP_SCHEDULE": "default",
            },
            num_stages=s,
            num_warps=w,
            num_buffers_warp_spec=0,
            num_consumer_groups=0,
        )
        if has_warp_spec
        else triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "ENABLE_TMA": False,
                "LOOP_SCHEDULE": "default",
            },
            num_stages=s,
            num_warps=w,
        )
    )
    for BM in [64, 128]
    for BN in [64, 128]
    for s in [3, 4, 7]
    for w in [4, 8]
]
# TMA, WS, and CompPipe
configsTmaWS = [
    (
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "ENABLE_TMA": enable_tma,
                "LOOP_SCHEDULE": sched,
            },
            num_stages=2 if sched == "FA_firstDot" or sched == "FA_secondDot" else 0,
            num_warps=w,
            num_buffers_warp_spec=buf,
            num_consumer_groups=grp,
            reg_dec_producer=dec,
            reg_inc_consumer=inc,
        )
        if has_warp_spec
        else triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "ENABLE_TMA": enable_tma,
                "LOOP_SCHEDULE": sched,
            },
            num_stages=2 if sched == "FA_firstDot" or sched == "FA_secondDot" else 0,
            num_warps=w,
        )
    )
    for BM in [128]
    for BN in [128]
    for sched in schedList
    for enable_tma in [True]
    for enable_ws in [True]
    for w in [4]
    for buf in [2]
    for grp in [2]  # 2
    for dec, inc in [
        (24, 240)
    ]  # , (40, 232)] #32,240 hangs, 24, 240 works 40, 232 works
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.jit
def _attn_fwd_compute(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,  #
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    Z,
    H,
    N_CTX,  #: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    K_block_ptr = None
    V_block_ptr = None
    Q_block_ptr = None
    O_block_ptr = None
    if not ENABLE_TMA:
        # block pointers
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=v_order,
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(HEAD_DIM, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    if ENABLE_TMA:
        q = tl._experimental_descriptor_load(  # load in row major
            desc_q,
            [(qvk_offset // stride_qm + start_m * BLOCK_M).to(tl.int32), 0],
            [BLOCK_M, HEAD_DIM],
            Q.dtype.element_ty,
        )
    else:
        q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,  #
            desc_k,
            desc_v,
            Q,
            qvk_offset,
            stride_kn,
            stride_vn,
            stride_vk,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,  #
            ENABLE_TMA,
            LOOP_SCHEDULE,
        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,  #
            desc_k,
            desc_v,
            Q,
            qvk_offset,
            stride_kn,
            stride_vn,
            stride_vk,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,  #
            ENABLE_TMA,
            LOOP_SCHEDULE,
        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    if ENABLE_TMA:
        tl._experimental_descriptor_store(
            desc_o,
            acc.to(Out.type.element_ty),
            [(qvk_offset // stride_om + start_m * BLOCK_M).to(tl.int32), 0],
        )
    else:
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _attn_fwd_compute_ws(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,  #
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    Z,
    H,
    N_CTX,  #: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    K_block_ptr = None
    V_block_ptr = None
    Q_block_ptr = None
    O_block_ptr = None
    if not ENABLE_TMA:
        # block pointers
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=v_order,
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(HEAD_DIM, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    with tl.async_task([0]):
        if ENABLE_TMA:
            q = tl._experimental_descriptor_load(  # load in row major
                desc_q,
                [(qvk_offset // stride_qm + start_m * BLOCK_M).to(tl.int32), 0],
                [BLOCK_M, HEAD_DIM],
                Q.dtype.element_ty,
            )
        else:
            q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner_ws(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,  #
            desc_k,
            desc_v,
            Q,
            qvk_offset,
            stride_kn,
            stride_vn,
            stride_vk,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,  #
            ENABLE_TMA,
            LOOP_SCHEDULE,
        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner_ws(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,  #
            desc_k,
            desc_v,
            Q,
            qvk_offset,
            stride_kn,
            stride_vn,
            stride_vk,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,  #
            ENABLE_TMA,
            LOOP_SCHEDULE,
        )
    # epilogue
    with tl.async_task([1, 2]):
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        if ENABLE_TMA:
            tl._experimental_descriptor_store(
                desc_o,
                acc.to(Out.type.element_ty),
                [(qvk_offset // stride_om + start_m * BLOCK_M).to(tl.int32), 0],
            )
        else:
            tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.autotune(list(filter(keep, configsWS)), key=["N_CTX"])
@triton.jit
def _attn_fwd_ws(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,  #
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    Z,
    H,
    N_CTX,  #: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
    ENABLE_WS: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    _attn_fwd_compute_ws(
        Q,
        K,
        V,
        sm_scale,
        M,
        Out,  #
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,  #
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,  #
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,  #
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,  #
        Z,
        H,
        N_CTX,  #: tl.constexpr,  #
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        STAGE,
        ENABLE_TMA,
        LOOP_SCHEDULE,
    )


@triton.autotune(list(filter(keep, configsOrig)), key=["N_CTX"])
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,  #
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    Z,
    H,
    N_CTX,  #: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
    ENABLE_WS: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    _attn_fwd_compute(
        Q,
        K,
        V,
        sm_scale,
        M,
        Out,  #
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,  #
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,  #
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,  #
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,  #
        Z,
        H,
        N_CTX,  #: tl.constexpr,  #
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        STAGE,
        ENABLE_TMA,
        LOOP_SCHEDULE,
    )


@triton.autotune(list(filter(keep, configsOpt)), key=["N_CTX"])
@triton.jit
def _attn_fwd_opt(  # Q, V, desc_k, desc_v, sm_scale, M, Out,  #
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,  #
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    Z,
    H,
    N_CTX,  #: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
    ENABLE_WS: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    _attn_fwd_compute(
        Q,
        K,
        V,
        sm_scale,
        M,
        Out,  #
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,  #
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,  #
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,  #
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,  #
        Z,
        H,
        N_CTX,  #: tl.constexpr,  #
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        STAGE,
        ENABLE_TMA,
        LOOP_SCHEDULE,
    )


@triton.autotune(list(filter(keep, configsTma)), key=["N_CTX"])
@triton.jit
def _attn_fwd_tma(  # Q, V, desc_k, desc_v, sm_scale, M, Out,  #
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,  #
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    Z,
    H,
    N_CTX,  #: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
    ENABLE_WS: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    _attn_fwd_compute(
        Q,
        K,
        V,
        sm_scale,
        M,
        Out,  #
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,  #
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,  #
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,  #
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,  #
        Z,
        H,
        N_CTX,  #: tl.constexpr,  #
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        STAGE,
        ENABLE_TMA,
        LOOP_SCHEDULE,
    )


@triton.autotune(list(filter(keep, configsTmaWS)), key=["N_CTX"])
@triton.jit
def _attn_fwd_tma_ws(  # Q, V, desc_k, desc_v, sm_scale, M, Out,  #
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,  #
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    Z,
    H,
    N_CTX,  #: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
    ENABLE_WS: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    _attn_fwd_compute_ws(
        Q,
        K,
        V,
        sm_scale,
        M,
        Out,  #
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,  #
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,  #
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,  #
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,  #
        Z,
        H,
        N_CTX,  #: tl.constexpr,  #
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        STAGE,
        ENABLE_TMA,
        LOOP_SCHEDULE,
    )


@triton.jit
def _attn_bwd_preprocess(
    O,
    DO,
    Delta,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #  #  #  #
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(
        O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    )
    do = tl.load(
        DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,  #
    Q,
    k,
    v,
    sm_scale,  #
    DO,  #
    M,
    D,  #
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    # Filled in by the wrapper.
    start_n,
    start_m,
    num_steps,  #
    MASK: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.bfloat16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.bfloat16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    K,
    V,  #
    do,
    m,
    D,
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    N_CTX,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    # Filled in by the wrapper.
    start_m,
    start_n,
    num_steps,  #
    MASK: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.bfloat16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,  #
    DO,  #
    DQ,
    DK,
    DV,  #
    M,
    D,
    # shared by Q/K/V/DO.
    stride_z,
    stride_h,
    stride_tok,
    stride_d,  #
    H,
    N_CTX,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    dk, dv = _attn_bwd_dkdv(
        dk,
        dv,  #
        Q,
        k,
        v,
        sm_scale,  #
        DO,  #
        M,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        MASK_BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
        MASK=True,  #
    )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(  #
        dk,
        dv,  #
        Q,
        k,
        v,
        sm_scale,  #
        DO,  #
        M,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
        MASK=False,  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,  #
        do,
        m,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M2,
        MASK_BLOCK_N2,
        HEAD_DIM,  #
        start_m,
        end_n - num_steps * MASK_BLOCK_N2,
        num_steps,  #
        MASK=True,  #
    )
    end_n -= num_steps * MASK_BLOCK_N2
    # stage 2
    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,  #
        do,
        m,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,  #
        start_m,
        end_n - num_steps * BLOCK_N2,
        num_steps,  #
        MASK=False,  #
    )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


class _attention_opt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, baseVariant):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-2] if v.dtype == torch.float8_e5m2 else v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}

        TMA_SIZE = 128
        BATCH, H, N_CTX = q.shape[0], q.shape[1], q.shape[2]
        # no autotune with fixed BLOCK_N
        desc_helper = TmaAutoTuneHelper()
        desc_helper.init_tma_descriptor("k")
        desc_helper.init_tma_descriptor("v")
        desc_helper.init_tma_descriptor("q")
        desc_helper.init_tma_descriptor("o")

        def grid_tma(META):
            if META["ENABLE_TMA"] == False:
                return (
                    # grid partitioning: num_consumer_groups * BLOCK_M
                    # data partitioning: BLOCK_M
                    triton.cdiv(q.shape[2], META["BLOCK_M"]),  # num_consumer_groups
                    q.shape[0] * q.shape[1],
                    1,
                )
            nonlocal desc_helper
            desc_helper.fill_2d_tma_descriptor(
                "k",
                k.data_ptr(),
                BATCH * H * N_CTX,
                HEAD_DIM_Q,
                META["BLOCK_N"],
                HEAD_DIM_Q,
                k.element_size(),
            )
            if v.dtype == torch.float8_e5m2:
                desc_helper.fill_2d_tma_descriptor(
                    "v",
                    v.data_ptr(),
                    BATCH * H * HEAD_DIM_Q,
                    N_CTX,
                    HEAD_DIM_Q,
                    META["BLOCK_N"],
                    v.element_size(),
                )
            else:
                desc_helper.fill_2d_tma_descriptor(
                    "v",
                    v.data_ptr(),
                    BATCH * H * N_CTX,
                    HEAD_DIM_Q,
                    META["BLOCK_N"],
                    HEAD_DIM_Q,
                    v.element_size(),
                )
            desc_helper.fill_2d_tma_descriptor(
                "q",
                q.data_ptr(),
                BATCH * H * N_CTX,
                HEAD_DIM_Q,
                META["BLOCK_M"]
                // (2 if META["ENABLE_WS"] else 1),  # data partitioning: halve
                HEAD_DIM_Q,
                q.element_size(),
            )
            desc_helper.fill_2d_tma_descriptor(
                "o",
                o.data_ptr(),
                BATCH * H * N_CTX,
                HEAD_DIM_Q,
                META["BLOCK_M"]
                // (2 if META["ENABLE_WS"] else 1),  # data partitioning: halve
                HEAD_DIM_Q,
                o.element_size(),
            )
            return (
                # grid partitioning: num_consumer_groups * BLOCK_M
                # data partitioning: BLOCK_M
                triton.cdiv(q.shape[2], META["BLOCK_M"]),  # num_consumer_groups
                q.shape[0] * q.shape[1],
                1,
            )

        desc_q = desc_helper.get_tma_descriptor_kernel_param("q")
        desc_k = desc_helper.get_tma_descriptor_kernel_param("k")
        desc_v = desc_helper.get_tma_descriptor_kernel_param("v")
        desc_o = desc_helper.get_tma_descriptor_kernel_param("o")

        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        if baseVariant == "base":
            _attn_fwd[grid_tma](
                q,
                k,
                v,
                sm_scale,
                M,
                o,
                desc_q,
                desc_k,
                desc_v,
                desc_o,  #
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),  #
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),  #
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),  #
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),  #
                q.shape[0],
                q.shape[1],  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                STAGE=stage,  #
                ENABLE_WS=False,
                **extra_kern_args,
            )
        elif baseVariant == "ws":
            _attn_fwd_ws[grid_tma](
                q,
                k,
                v,
                sm_scale,
                M,
                o,
                desc_q,
                desc_k,
                desc_v,
                desc_o,  #
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),  #
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),  #
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),  #
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),  #
                q.shape[0],
                q.shape[1],  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                STAGE=stage,  #
                ENABLE_WS=True,
                **extra_kern_args,
            )
        elif baseVariant == "opt":
            _attn_fwd_opt[grid_tma](
                q,
                k,
                v,
                sm_scale,
                M,
                o,
                desc_q,
                desc_k,
                desc_v,
                desc_o,  #
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),  #
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),  #
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),  #
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),  #
                q.shape[0],
                q.shape[1],  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                STAGE=stage,  #
                ENABLE_WS=False,
                **extra_kern_args,
            )
        elif baseVariant == "tma":
            _attn_fwd_tma[grid_tma](
                q,
                k,
                v,
                sm_scale,
                M,
                o,
                desc_q,
                desc_k,
                desc_v,
                desc_o,  #
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),  #
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),  #
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),  #
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),  #
                q.shape[0],
                q.shape[1],  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                STAGE=stage,  #
                ENABLE_WS=False,
                **extra_kern_args,
            )
        elif baseVariant == "tma_ws":
            _attn_fwd_tma_ws[grid_tma](
                q,
                k,
                v,
                sm_scale,
                M,
                o,
                desc_q,
                desc_k,
                desc_v,
                desc_o,  #
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),  #
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),  #
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),  #
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),  #
                q.shape[0],
                q.shape[1],  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                STAGE=stage,  #
                ENABLE_WS=True,
                **extra_kern_args,
            )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid_tma
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o,
            do,  #
            delta,  #
            BATCH,
            N_HEAD,
            N_CTX,  #
            BLOCK_M=PRE_BLOCK,
            HEAD_DIM=ctx.HEAD_DIM,  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            do,
            dq,
            dk,
            dv,  #
            M,
            delta,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            N_HEAD,
            N_CTX,  #
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES,  #
        )

        return dq, dk, dv, None, None


attention_opt = _attention_opt.apply
