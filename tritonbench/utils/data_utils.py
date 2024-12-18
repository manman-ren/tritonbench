from .triton_op import IS_FBCODE


def get_production_shapes(op_name, op_type, shuffle_shapes=False):
    """Gets a list of Softmax shapes for benchmarking"""
    if IS_FBCODE:
        from .fb.durin_data import productionDataLoader

        return [
            shape
            for shape in productionDataLoader.get_shapes_from_frozen_durin(
                op_name, op_type, shuffle_shapes
            )
        ]
