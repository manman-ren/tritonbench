import torch
import triton
import triton.language as tl
from tritonbench.utils.path_utils import add_path, SUBMODULE_PATH

try:
    from hammer.ops.triton.utils import prev_power_of_2

    # Internal Import
    from hammer.oss.generative_recommenders.ops.triton.triton_ragged_hstu_attention import (
        _ragged_hstu_attn_fwd,
        _ragged_hstu_attn_fwd_persistent,
        _ragged_hstu_attn_fwd_tma,
        _ragged_hstu_attn_fwd_ws,
    )
except ModuleNotFoundError:
    # OSS Import
    import importlib

    with add_path(str(SUBMODULE_PATH)):
        triton_ragged_hstu_attention = importlib.import_module(
            "generative-recommenders.ops.triton.triton_ragged_hstu_attention"
        )
        _ragged_hstu_attn_fwd = triton_ragged_hstu_attention._ragged_hstu_attn_fwd
        _ragged_hstu_attn_fwd_persistent = (
            triton_ragged_hstu_attention._ragged_hstu_attn_fwd_persistent
        )
        _ragged_hstu_attn_fwd_tma = (
            triton_ragged_hstu_attention._ragged_hstu_attn_fwd_tma
        )
        _ragged_hstu_attn_fwd_ws = (
            triton_ragged_hstu_attention._ragged_hstu_attn_fwd_ws
        )

    @torch.fx.wrap
    def prev_power_of_2(x: int) -> int:
        if torch.compiler.is_compiling():
            # Re-write to make Dynamo happy
            x_tensor = torch.scalar_tensor(x, dtype=torch.int64)  # type: ignore[arg-type]
            x_tensor_orig = x_tensor.clone()
            out = triton.next_power_of_2(x_tensor)  # type: ignore[arg-type]
            return int(torch.where(torch.lt(x_tensor_orig, out), out // 2, out).item())  # type: ignore[return-value]
        else:
            out = triton.next_power_of_2(x)
            return out // 2 if out > x else out


from typing import Tuple
# check if we have the TMA version in Triton PR #4498 (https://github.com/triton-lang/triton/pull/4498).
HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)

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


class RaggedHSTUAttn(torch.nn.Module):
    def __init__(
        self,
        batch_size,
        num_heads,
        max_seq_len,
        num_buckets,
        persistent_kernel: bool = False,
        enable_tma: bool = False,
        enable_ws: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.all_ts_weights = torch.nn.Parameter(
            torch.randn(
                (self.num_buckets + 1,),
                dtype=torch.bfloat16,
            ).cuda()
        )
        self.all_pos_weights = torch.nn.Parameter(
            torch.randn(
                (2 * self.max_seq_len - 1,),
                dtype=torch.bfloat16,
            ).cuda()
        )
        self.persistent_kernel = persistent_kernel
        self.enable_tma = enable_tma
        self.enable_ws = enable_ws

    def forward(
        self, qkv: torch.Tensor, seq_offsets: torch.Tensor, timestamps: torch.Tensor
    ) -> torch.Tensor:
        NUM_BUCKETS = self.num_buckets
        torch._check(timestamps.size(0) + 1 == seq_offsets.size(0))

        q = qkv[:, :, :128]
        k = qkv[:, :, 128:256]
        v = qkv[:, :, 256:384]
        q = q.contiguous()  # [batch x seqlen] x head x dim
        k = k.contiguous()
        v = v.contiguous()

        out = torch.zeros_like(v)

        Z = timestamps.size(0)
        N = timestamps.size(1) - 1
        L, H, DimQ = q.shape
        _, _, DimV = v.shape

        # set up descriptors for TMA
        TMA_SIZE = 128
        BLOCK_D_V, BLOCK_D_Q = DimV, DimQ
        desc_helper = TmaAutoTuneHelper()
        desc_helper.init_tma_descriptor("k")
        desc_helper.init_tma_descriptor("v")
        desc_helper.init_tma_descriptor("q")
        desc_helper.init_tma_descriptor("o")

        def grid_tma(META):
            if self.enable_tma == False:
                return (  # noqa E731
                    triton.cdiv(N, META["BLOCK_M"]),
                    Z * H, 1,
                )

            nonlocal desc_helper
            desc_helper.fill_2d_tma_descriptor(
                "k",
                k.data_ptr(),
                L, H * DimQ,
                META["BLOCK_N"],
                BLOCK_D_Q,
                k.element_size(),
            )
            desc_helper.fill_2d_tma_descriptor(
                "v",
                v.data_ptr(),
                L, H * DimV,
                META["BLOCK_N"],
                BLOCK_D_V,
                v.element_size(),
            )
            desc_helper.fill_2d_tma_descriptor(
                "q",
                q.data_ptr(),
                L, H * DimQ,
                META["BLOCK_M"] // 2, # data partitioning
                BLOCK_D_Q,
                q.element_size(),
            )
            desc_helper.fill_2d_tma_descriptor(
                "o",
                out.data_ptr(),
                L, H * DimV,
                META["BLOCK_M"] // 2, # data partitioning
                BLOCK_D_V,
                out.element_size(),
            )
            return (
                triton.cdiv(N, META["BLOCK_M"]),
                Z * H,
                1,
            )

        desc_q = desc_helper.get_tma_descriptor_kernel_param("q")
        desc_k = desc_helper.get_tma_descriptor_kernel_param("k")
        desc_v = desc_helper.get_tma_descriptor_kernel_param("v")
        desc_o = desc_helper.get_tma_descriptor_kernel_param("o")
        assert k.stride(1) == v.stride(1) #stride_vh

        kwargs = {
            "Q": q,
            "K": k,
            "V": v,
            "desc_q": desc_q,
            "desc_k": desc_k,
            "desc_v": desc_v,
            "desc_o": desc_o,
            "seq_offsets": seq_offsets,
            "delta_x_offsets": None,
            "TS": timestamps,
            "TW": self.all_ts_weights,
            "PW": self.all_pos_weights,
            "Bias": None,
            "seq2_offsets": None,
            "num_targets": None,
            "Scale": None,
            "Out": out,
            "stride_qm": q.stride(0),
            "stride_qh": q.stride(1),
            "stride_kn": k.stride(0),
            "stride_kh": k.stride(1),
            "stride_vn": v.stride(0),
            "stride_vh": v.stride(1),
            "stride_sz": None,
            "stride_sm": None,
            "stride_ts": timestamps.stride(0),
            "stride_om": out.stride(0),
            "stride_oh": out.stride(1),
            "alpha": 0.08838834764831843,
            "Z": Z,
            "H": H,
            "MAX_SEQ_LEN": N,
            "AUTOTUNE_MAX_SEQ_LEN": prev_power_of_2(N),
            "DimQ": DimQ,
            "DimV": DimV,
            "DeltaSize": None,
            "num_buckets": NUM_BUCKETS,
            "max_pos_ind": None,
            "time_bucket_incr": 60.0,
            "time_bucket_div": 1.0,
            "time_delta": 0.0,
            "INVALID_MASK_TYPE": "lower_triangular",
            "CAUSAL": True,
            "BUCKET_FN": "sqrt",
            "ATTN_BIAS_TYPE": "fused",
            "USE_TIME_BIAS": False,
            "USE_POS_BIAS": False,
            "HAS_MAX_POS_IND": False,
            "HAS_MULTIPLE_TARGETS": False,
            "HAS_ATTN_SCALE": False,
            "IS_DELTA_Q": False,
            "ALLOW_TF32": True,
            "BLOCK_D_Q": DimQ,
            "BLOCK_D_V": DimV,
            "max_attn_len": 0,
            "HAS_CONTEXTUAL_SEQ_LEN": False,
            "contextual_seq_len": 0,
            "HAS_SORT_BY_LENGTH_INDICES": False,
            "sort_by_length_indices": None,
            "enable_tma": self.enable_tma,
        }
        if self.persistent_kernel:
            grid = (1216,)
            _ragged_hstu_attn_fwd_persistent[grid](**kwargs)
        else:
            if self.enable_tma:
                _ragged_hstu_attn_fwd_tma[grid_tma](**kwargs)
            elif self.enable_ws:
                _ragged_hstu_attn_fwd_ws[grid_tma](**kwargs)
            else:
                _ragged_hstu_attn_fwd[grid_tma](**kwargs)

        return out


def get_test_inputs(
    batch_size, num_heads, max_seq_len
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    timestamp_deltas: torch.Tensor = (
        torch.randint(
            86400,
            size=(batch_size, max_seq_len + 1),
        )
        .requires_grad_(False)
        .cuda()
    )
    timestamps = timestamp_deltas.cumsum(dim=1)

    # sparsity >= 0.5
    sparsity = 0.8
    min_seq_len: int = int((2 * sparsity - 1.0) * max_seq_len)
    lengths = (
        torch.randint(
            low=min_seq_len, high=max_seq_len, #max_seq_len - 10, max_seq_len + 1,
            size=(batch_size,),
        )
        .requires_grad_(False)
        .cuda()
    )
    seq_offsets = (
        torch.zeros(
            (batch_size + 1,),
            dtype=torch.int64,
        )
        .requires_grad_(False)
        .cuda()
    )
    seq_offsets[1:] = torch.cumsum(
        lengths,
        dim=0,
    )
    L = int(seq_offsets[-1].item())

    qkv = (
        torch.randn(
            (L, num_heads, 512),
            dtype=torch.bfloat16,
        )
        .requires_grad_(False)
        .cuda()
    )
    return qkv, seq_offsets, timestamps
