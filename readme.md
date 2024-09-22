<div align="center">

# ComfyUI Facerestore TensorRT

[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-12.4-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-10.4-green)](https://developer.nvidia.com/tensorrt)
[![by-nc-sa/4.0](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

</div>

<p align="center">
  <img src="assets/demo.PNG" />

</p>

This project provides an experimental Tensorrt implementation for ultra fast face restoration inside ComfyUI.

Note: This project doesn't do pre/post processing. It only works on cropped faces for now.



If you like the project, please give sa star! ⭐

---

## ⏱️ Performance

_Note: The following results were benchmarked  ComfyUI, using 100 similar frames_

| Device |  MODEL | PRECISION| FPS |
|---------|--------|---|---|
|  RTX 3090  | Codeformer  | FP16| 15.6|
|  RTX 3090  | Gfqgan  | FP32| 13.1|

## 🚀 Installation

Navigate to the ComfyUI `/custom_nodes` directory

```bash
git clone https://github.com/yuvraj108c/ComfyUI-Facerestore-Tensorrt
cd ./ComfyUI-Facerestore-Tensorrt
pip install -r requirements.txt
```

## 🛠️ Building Tensorrt Engine

1. Download one of the following onnx models:
   - [gfqgan.onnx](https://huggingface.co/yuvraj108c/facerestore-onnx/resolve/main/gfqgan.onnx)
   - [codeformer.onnx](https://huggingface.co/yuvraj108c/facerestore-onnx/resolve/main/codeformer.onnx)
2. Build tensorrt engines for these models by running:

   - `python export_trt.py`

3. Place the exported engines inside ComfyUI `/models/tensorrt/facerestore` directory

## ☀️ Usage

- Insert node by `Right Click -> tensorrt -> Face Restore Tensorrt`

## 🤖 Environment tested

- Ubuntu 22.04 LTS, Cuda 12.4, Tensorrt 10.4.0, Python 3.10, RTX 3090 GPU
- Windows (Not tested, but should work)

## 👏 Credits

- https://github.com/bychen7/Face-Restoration-TensorRT
- https://github.com/yuvraj108c/Codeformer-Tensorrt

## License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)