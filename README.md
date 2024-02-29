# TensorRT-FastSAM
![](assets/logo.png)

## Requirements
```sh
pip install -r requirements.txt
```

## Export ONNX model
```sh
python export.py --weights < path to FastSAM-x.pt > --output <onnx_model_path>
##--------- example ---------- 
# python export.py --weights ../models/FastSAM-x.pt --output ../models/FastSAM-x2.onnx
```

## Export trt engine
```sh
/usr/src/tensorrt/bin/trtexec \ 
--onnx=<onnx_model_path> \
--saveEngine=<engine_path> \
--explicitBatch \ 
--minShapes=images:1x3x1024x1024 \
--optShapes=images:1x3x1024x1024 \
--maxShapes=images:4x3x1024x1024 \ 
--verbose \ 
--device=0
##--------- example ---------- 
# /usr/src/tensorrt/bin/trtexec \
# --onnx="../models/FastSAM-x2.onnx" \
# --saveEngine="../trt_engines/FastSAM-x.engine" \
# --explicitBatch \
# --minShapes=images:1x3x1024x1024 \
# --optShapes=images:1x3x1024x1024 \
# --maxShapes=images:4x3x1024x1024 \
# --verbose \
# --device=0
```

## Do inference

```sh
python trt_infer.py <engine_path> <image_path> --output=<output_path>
##--------- example ---------- 
# python trt_infer.py "../trt_engines/FastSAM-x.engine" --output="../unnamed.jpg"
```

## References

- [TensorRT](https://github.com/NVIDIA/TensorRT)
- [FastSam_Awsome_TensorRT](https://github.com/ChuRuaNh0/FastSam_Awsome_TensorRT)