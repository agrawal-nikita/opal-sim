# SPDX-License-Identifier: Apache-2.0
STR_DTYPE_TO_BYTES = {
    "half": 2,  # torch.float16 / torch.half
    "bfloat16": 2,  # torch.bfloat16
    "float": 4,  # torch.float32
    "double": 8,  # torch.float64
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "bool": 1,  # PyTorch uses 1 byte for bool
}
