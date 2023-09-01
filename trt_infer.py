import time
import numpy as np
import tensorrt as trt
from cuda import cudart
import time
import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results
from ultralytics.yolo.utils import ops
from PIL import Image
from random import randint

import common
from utils import overlay

class TensorRTInfer:
    """
    Implements inference for  TensorRT engine.
    """

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            shape = self.context.get_binding_shape(i)
            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            # print("{} '{}' with shape {} and dtype {}".format(
            #     "Input" if is_input else "Output",
            #     binding['name'], binding['shape'], binding['dtype']))

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, batch):
        """
        Execute inference on a batch of images.
        :param batch: A numpy array holding the image batch.
        :return A list of outputs as numpy arrays.
        """
        # Copy I/O and Execute
        common.memcpy_host_to_device(self.inputs[0]['allocation'], batch)
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            common.memcpy_device_to_host(self.outputs[o]['host_allocation'], self.outputs[o]['allocation'])
        return [o['host_allocation'] for o in self.outputs]


def postprocess(preds, img, orig_imgs, retina_masks, conf, iou, agnostic_nms=True):
    """TODO: filter by classes."""

    p = ops.non_max_suppression(preds[0],
                                conf,
                                iou,
                                agnostic_nms,
                                max_det=100,
                                nc=1)
    results = []
    proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
    for i, pred in enumerate(p):
        orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        # path = self.batch[0]
        img_path = "ok"
        if not len(pred):  # save empty boxes
            results.append(Results(orig_img=orig_img, path=img_path, names="segment", boxes=pred[:, :6]))
            continue
        if retina_masks:
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
        else:
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results.append(
            Results(orig_img=orig_img, path=img_path, names="1213", boxes=pred[:, :6], masks=masks))
    return results


def pre_processing(img_origin, imgsz=1024):
    h, w = img_origin.shape[:2]
    if h > w:
        scale = min(imgsz / h, imgsz / w)
        inp = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        nw = int(w * scale)
        nh = int(h * scale)
        a = int((nh - nw) / 2)
        inp[: nh, a:a + nw, :] = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
    else:
        scale = min(imgsz / h, imgsz / w)
        inp = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        nw = int(w * scale)
        nh = int(h * scale)
        a = int((nw - nh) / 2)

        inp[a: a + nh, :nw, :] = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
    rgb = np.array([inp], dtype=np.float32) / 255.0
    rgb = np.transpose(rgb, (0, 3, 1, 2))
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    return rgb


class FastSam(object):
    def __init__(self,
                 model_weights='fast_sam_1024.trt',
                 max_size=1024):
        self.imgsz = (max_size, max_size)
        # Load model
        self.model = TensorRTInfer(model_weights)

    def segment(self, bgr_img, retina_masks, conf, iou, agnostic_nms):
        ## Padded resize
        inp = pre_processing(bgr_img, self.imgsz[0])
        ## Inference
        print("[Input]: ", inp[0].transpose(0, 1, 2).shape)
        preds = self.model.infer(inp)
        data_0 = torch.from_numpy(preds[5])
        data_1 = [[torch.from_numpy(preds[2]), torch.from_numpy(preds[3]), torch.from_numpy(preds[4])],
                  torch.from_numpy(preds[1]), torch.from_numpy(preds[0])]
        preds = [data_0, data_1]
        print(inp.shape, bgr_img.shape, retina_masks)
        result = postprocess(preds, inp, bgr_img, retina_masks, conf, iou, agnostic_nms)
        masks = result[0].masks.data
        print("len of mask: ", len(masks))
        image_with_masks = np.copy(bgr_img)
        for i, mask_i in enumerate(masks):
            r = randint(0, 255)
            g = randint(0, 255)
            b = randint(0, 255)
            rand_color = (r, g, b)
            image_with_masks = overlay(image_with_masks, mask_i, color=rand_color, alpha=1)
        cv2.imwrite("obj_segment_trt.png", image_with_masks)

        return masks

    def batch_segment(self, img_list, retina_masks, conf, iou, agnostic_nms):
        ## Padded resize
        tenosr = []
        org = []
        for path in img_list:
            bgr_img = cv2.imread(path)
            org.append(bgr_img)
            inp = pre_processing(bgr_img, self.imgsz[0])
            tenosr.append(inp)
        inp = np.concatenate(tenosr, axis=0)
        ## Inference
        print("[Input]: ", inp[0].transpose(0, 1, 2).shape)
        preds = self.model.infer(inp)
        data_0 = torch.from_numpy(preds[5])
        data_1 = [[torch.from_numpy(preds[2]), torch.from_numpy(preds[3]), torch.from_numpy(preds[4])],
                  torch.from_numpy(preds[1]), torch.from_numpy(preds[0])]
        preds = [data_0, data_1]
        print(inp.shape, tenosr[0].shape, retina_masks)
        results = postprocess(preds, inp, org[0], retina_masks, conf, iou, agnostic_nms)
        
        for index, result in enumerate(results):
            masks = result.masks.data
            print("len of mask: ", len(masks))
            image_with_masks = np.copy(org[index])
            for i, mask_i in enumerate(masks):
                r = randint(0, 255)
                g = randint(0, 255)
                b = randint(0, 255)
                rand_color = (r, g, b)
                image_with_masks = overlay(image_with_masks, mask_i, color=rand_color, alpha=1)
            image_with_masks = np.hstack([org[index], image_with_masks])
            cv2.imwrite(f"{index}_obj_segment_trt.png", image_with_masks)

        return masks

if __name__ == '__main__':
    retina_masks = True
    conf = 0.1
    iou = 0.25
    agnostic_nms = False

    model = FastSam(model_weights="FastSAM-x.trt")
    # single inference
    img = cv2.imread('xxx.png')
    masks = model.segment(img, retina_masks, conf, iou, agnostic_nms)

    #batch inference
    imgs = ['xx.bmp', 'xx.bmp',
             'xx.bmp', 'xx.bmp']
    masks = model.batch_segment(imgs, retina_masks, conf, iou, agnostic_nms)