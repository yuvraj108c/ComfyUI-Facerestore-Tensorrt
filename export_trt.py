import sys
import os
import torch
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from trt_utilities import Engine

def export_trt(trt_path: str, onnx_path: str, use_fp16: bool):
    if not os.path.exists(onnx_path):
        print(f"Skipping conversion: {onnx_path} not found.")
        return False
    
    try:
        engine = Engine(trt_path)
        torch.cuda.empty_cache()
        s = time.time()
        ret = engine.build(
            onnx_path,
            use_fp16,
            enable_preview=True,
        )
        e = time.time()
        print(f"Time taken to build: {(e-s)} seconds")
        return ret
    except Exception as e:
        print(f"Error converting {onnx_path}: {str(e)}")
        return False

# List of models to convert
models = [
    {"trt_path": "./codeformer.engine", "onnx_path": "./codeformer.onnx", "use_fp16": True},
    {"trt_path": "./gfqgan.engine", "onnx_path": "./gfqgan.onnx", "use_fp16": False},
    {"trt_path": "./gfpgan14.engine", "onnx_path": "./GFPGANv1.4.onnx", "use_fp16": False},
    #{"trt_path": "./gpenbfr2048.engine", "onnx_path": "./GPEN-BFR-2048.onnx", "use_fp16": False}, - fails, boo
]

for model in models:
    result = export_trt(**model)
    if result:
        print(f"Successfully converted {model['onnx_path']}")
    else:
        print(f"Failed to convert {model['onnx_path']}")
