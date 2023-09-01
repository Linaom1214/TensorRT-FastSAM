# TensorRT-FastSAM
![](assets/logo.png)
## Export ONNX model
```sheel
python export.py --weights FastSAM-x.pt --outputFastSAM-x.onnx
```
## Export trt engine
```sheel
./trtexec --onnx=./FastSAM-s.onnx --saveEngine=./FastSAM-s.trt --explicitBatch --minShapes=images:1x3x1024x1024 --optShapes=images:1x3x1024x1024 --maxShapes=images:4x3x1024x1024 --verbose --device=0
```
## Do inference

```sheel
python trt_infer.py
```

## Reference

## [TensorRT](https://github.com/NVIDIA/TensorRT)、[FastSam_Awsome_TensorRT](https://github.com/ChuRuaNh0/FastSam_Awsome_TensorRT)