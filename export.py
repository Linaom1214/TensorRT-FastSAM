from ultralytics import YOLO
import argparse
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./FastSAM-x.pt', help='weights path')
    parser.add_argument('--output', type=str, default='FastSAM-x.onnx', help='output ONNX model path')
    parser.add_argument('--max_size', type=int, default=416, help='max size of input image')
    opt = parser.parse_args()

    model_weights = opt.weights
    output_model_path = opt.output
    max_size = opt.max_size
    device = torch.device("cuda")

    # load model 
    print("[Info] Load Model")
    model_ = YOLO(model_weights)
    model = model_.model

    img = torch.zeros(1, 3, max_size, max_size).to(device)

    print("[Info] Preprocess Model")
    output_names = ['output0', 'output1'] #if isinstance(model, SegmentationModel) else ['output0']
    dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
    dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
    dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)

    model.eval().to(device)

    print('[INFO] Convert from Torch to ONNX')

    torch.onnx.export(model,               # model being run
                    img,                         # model input (or a tuple for multiple inputs)
                    output_model_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['images'],   # the model's input names
                    output_names = output_names, # the model's output names
                    dynamic_axes=dynamic)

    print('[INFO] Finished Convert!')