import os
import ctypes
import time
import sys
import argparse

import cv2
import numpy as np
from PIL import Image
import tensorrt as trt
from utils.engine import build_engine_onnx, get_engine, build_engine
from utils.general import non_max_suppression_np
from utils import common

import utils.inference as inference_utils # TRT/TF inference wrappers
import utils.model as model_utils # UFF conversion
import utils.boxes as boxes_utils # Drawing bounding boxes
import utils.coco as coco_utils # COCO dataset descriptors
from utils.paths import PATHS # Path management
from utils.datasets import *
from utils.general import *
from utils.torch_utils import select_device, time_sync
import pycuda.driver as cuda
import pycuda.autoinit
from utils.plots import *
from utils.metrics import box_iou
from utils.calibrator import Yolov5EntropyCalibrator
# COCO label list
COCO_LABELS = coco_utils.COCO_CLASSES_LIST

# Model used for inference
MODEL_NAME = 'yolov5l'

# Confidence threshold for drawing bounding box
VISUALIZATION_THRESHOLD = 0.5

# Precision command line argument -> TRT Engine datatype
TRT_PRECISION_TO_DATATYPE = {
    16: trt.DataType.HALF,
    32: trt.DataType.FLOAT,
    8: trt.DataType.INT8
}

# Layout of TensorRT network output metadata
TRT_PREDICTION_LAYOUT = {
    "image_id": 0,
    "label": 1,
    "confidence": 2,
    "xmin": 3,
    "ymin": 4,
    "xmax": 5,
    "ymax": 6
}



def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct




def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')



def save_one_image(predn, names, save_conf, shape, file, img_id, im0) :

    #im0 = cv2.imread('/home/youngjin/datasets/coco/val/images/' + img_id + '.jpg')
    annotator = Annotator(im0, line_width=3, example=str(names))
    for *xyxy, conf, cls in predn.tolist() :
        c = int(cls)
        label = names[c]
        annotator.box_label(xyxy, label, color=(c, True))
    im0 = annotator.result()
    cv2.imwrite(file, im0)














def fetch_prediction_field(field_name, detection_out, pred_start_idx):
    """Fetches prediction field from prediction byte array.
    After TensorRT inference, prediction data is saved in
    byte array and returned by object detection network.
    This byte array contains several pieces of data about
    prediction - we call one such piece a prediction field.
    The prediction fields layout is described in TRT_PREDICTION_LAYOUT.
    This function, given prediction byte array returned by network,
    staring index of given prediction and field name of interest,
    returns prediction field data corresponding to given arguments.
    Args:
        field_name (str): field of interest, one of keys of TRT_PREDICTION_LAYOUT
        detection_out (array): object detection network output
        pred_start_idx (int): start index of prediction of interest in detection_out
    Returns:
        Prediction field corresponding to given data.
    """
    return detection_out[pred_start_idx + TRT_PREDICTION_LAYOUT[field_name]]

def analyze_prediction(detection_out, pred_start_idx, img_pil):
    image_id = int(fetch_prediction_field("image_id", detection_out, pred_start_idx))
    label = int(fetch_prediction_field("label", detection_out, pred_start_idx))
    confidence = fetch_prediction_field("confidence", detection_out, pred_start_idx)
    xmin = fetch_prediction_field("xmin", detection_out, pred_start_idx)
    ymin = fetch_prediction_field("ymin", detection_out, pred_start_idx)
    xmax = fetch_prediction_field("xmax", detection_out, pred_start_idx)
    ymax = fetch_prediction_field("ymax", detection_out, pred_start_idx)
    if confidence > VISUALIZATION_THRESHOLD:
        class_name = COCO_LABELS[label]
        confidence_percentage = "{0:.0%}".format(confidence)
        print("Detected {} with confidence {}".format(
            class_name, confidence_percentage))
        boxes_utils.draw_bounding_boxes_on_image(
            img_pil, np.array([[ymin, xmin, ymax, xmax]]),
            display_str_list=["{}: {}".format(
                class_name, confidence_percentage)],
            color=coco_utils.COCO_COLORS[label]
        )

def parse_commandline_arguments():
    """Parses command line arguments and adjusts internal data structures."""

    # Define script command line arguments
    parser = argparse.ArgumentParser(description='Run object detection inference on input image.')
    parser.add_argument('--input_img_path', metavar='INPUT_IMG_PATH',default='/home/youngjin/datasets/coco/val',
        help='an image file to run inference on')
    parser.add_argument('-p', '--precision', type=int, choices=[32, 16, 8], default=8,
        help='desired TensorRT float precision to build an engine with')
    parser.add_argument('-b', '--max_batch_size', type=int, default=1,
        help='max TensorRT engine batch size')
    parser.add_argument('-w', '--workspace_dir',default='../runs/onnx_trt_detect',
        help='sample workspace directory')
    parser.add_argument('-fc', '--flatten_concat',
        help='path of built FlattenConcat plugin')
    parser.add_argument('-d', '--calib_dataset_path', default='/home/youngjin/datasets/coco/train/images',
        help='path to the calibration dataset')
    parser.add_argument('-c', '--calib_cache_path', default='/home/youngjin/projects/runs/calib_cache/coco_calib_cache.cache',
        help='path to the calibration dataset')
    parser.add_argument('--imgsz', default=(640,640))
    parser.add_argument('--workers', default=8)
    parser.add_argument('--device', default='0')
    parser.add_argument('--resize', default=True)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--stride', default=32)
    parser.add_argument('--conf_thres', default=0.45)
    parser.add_argument('--iou_thres', default=0.25)
    parser.add_argument('--data', default='./dataset/coco.yaml')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic_nms', default=False)
    parser.add_argument('--save_engine', default=False)
    parser.add_argument('--half', default=False)
    parser.add_argument('--int8', default=True)
    parser.add_argument('--evaluate', default=True)
    parser.add_argument('--save_img', default=True)
    parser.add_argument('--save_txt', default=True)
    parser.add_argument('--save_conf', default=False)
    parser.add_argument('--save_dir', default='/home/youngjin/projects/runs/onnx_trt_detect/')


    # Parse arguments passed
    args = parser.parse_args()

    # Set FlattenConcat TRT plugin path and
    # workspace dir path if passed by user
    if args.flatten_concat:
        PATHS.set_flatten_concat_plugin_path(args.flatten_concat)
    if args.workspace_dir:
        PATHS.set_workspace_dir_path(args.workspace_dir)

    try:
        os.makedirs(PATHS.get_workspace_dir_path())
    except:
        pass

    # Verify Paths after adjustments. This also exits script if verification fails
    PATHS.verify_all_paths()

    # Fetch TensorRT engine path and datatype
    args.trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[args.precision]
    args.trt_engine_path = PATHS.get_engine_path(args.trt_engine_datatype,
        args.max_batch_size)
    try:
        os.makedirs(os.path.dirname(args.trt_engine_path))
    except:
        pass

    return args

def main():

    # Parse command line arguments
    args = parse_commandline_arguments()

    device = select_device(args.device, batch_size=args.batch_size)


    # Directories
    save_dir = increment_path(Path(args.save_dir) / 'exp', exist_ok=False)  # increment run
    (save_dir / 'labels' if args.save_txt else args.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'images' if args.save_txt else args.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    yolo_model_onnx_path = '/home/youngjin/projects/yolov5l.onnx'
    #engine_path = '/home/youngjin/runs/onnx_trt_detect/models/yolov5/yolov5.trt'

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    # set shape
    profile_shape = (1, 3, args.imgsz[0], args.imgsz[1])

    profile = builder.create_optimization_profile()
    profile.set_shape("images", profile_shape, profile_shape, profile_shape)

    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    config.max_workspace_size = common.GiB(1)
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(yolo_model_onnx_path)):
        raise RuntimeError(f'failed to load ONNX file: {yolo_model_onnx_path}')

    # fp16 mode
    if builder.platform_has_fast_fp16 and args.half:
        config.set_flag(trt.BuilderFlag.FP16)

    # int8 mode 
    elif builder.platform_has_fast_int8 and args.int8 :
        config.set_flag(trt.BuilderFlag.INT8)

        calib = Yolov5EntropyCalibrator(args.calib_dataset_path, args.calib_cache_path, profile_shape)
        config.int8_calibrator = calib

    engine = builder.build_engine(network, config)
    context = engine.create_execution_context()

    dataset = create_dataloader_custom(image_path=args.input_img_path+'/images',
                                                    label_path=args.input_img_path+'/labels',
                                                    imgsz=args.imgsz,
                                                    batch_size=args.batch_size,
                                                    stride=args.stride,
                                                    workers=args.workers, 
                                                    resize=args.resize)

    data = check_dataset(args.data)
    names = data['names']
    nc = data['nc']
    eps = 1e-16
    pbar = tqdm(dataset)
    iouv = torch.linspace(0.5, 0.95, 10, device=device) 
    niou = iouv.numel()

    dt = [0.0, 0.0, 0.0]
    stats = []
    seen = 0
    for batch_i, (img, im0, targets, paths, shapes, img_id) in enumerate(pbar) :
        t0 = time_sync()
        img = img / 255
        img = img.astype(np.float32)

        stream = cuda.Stream()
        bindings = []


        for binding in engine : 

            if engine.binding_is_input(binding) : 
                context.set_binding_shape(0, profile_shape)
                shape = context.get_binding_shape(0)
                size = trt.volume(shape)
                dtype = trt.nptype(engine.get_binding_dtype(binding))

                host_mem = cuda.pagelocked_empty(size, dtype).reshape(shape)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                input = common.HostDeviceMem(host_mem, device_mem)
            else :

                size = trt.volume(engine.get_binding_shape(binding))
                dtype = trt.nptype(engine.get_binding_dtype(binding))

                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)


                output = common.HostDeviceMem(host_mem, device_mem)

            bindings.append(int(device_mem))

        np.copyto(input.host, img)
        t1 = time_sync()
        dt[0] += t1 - t0
        
        trt_outputs, dt[1] = common.do_inference_v2(context, bindings, input, output, stream, dt[1])

        t3 = time_sync()
        trt_outputs = trt_outputs.reshape((1, -1, nc + 5))
        outputs = non_max_suppression_np(trt_outputs, args.conf_thres, args.iou_thres, args.agnostic_nms)
        dt[2] += time_sync() - t3

        seen += 1


        if args.evaluate and (targets is not None):
            targets = torch.tensor(targets)
            for si, pred in enumerate(outputs) : 
                pred = torch.tensor(pred).to(device=device)
                pred = pred.view(-1, 6)
                cat_ids, bboxes = coco91_to_coco80_class(targets)
                nl, npr = cat_ids.shape[0], pred.shape[0]
                correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)

                if npr == 0 :
                    if nl : 
                        stats.append((correct, *torch.zeros((3, 0), device=device)))
                    continue
                
                predn = pred.clone()
                predn[:, :4] = scale_coords(img.shape[1:], predn[:, :4], im0.shape)

                if nl : 
                    bboxes = xywh2xyxy_custom2(bboxes)
                    labelsn = torch.cat((cat_ids, bboxes), 1)
                    correct = process_batch(predn, labelsn, iouv)

                stats.append((correct, pred[:, 4], pred[:, 5], cat_ids[:, 0]))

                if args.save_txt : 
                    save_one_txt(predn, args.save_conf, shapes, file= save_dir / 'labels' / str(img_id + '.txt'))

                if args.save_img :
                    save_one_image(predn, names, args.save_conf, shapes, 
                    file= save_dir / 'images' / str(img_id + '.jpg'), img_id=img_id, im0=im0)


    
    if args.evaluate :
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
        tp, _, _, target_cls = stats

        n_l = target_cls.shape[0]
        fpc = (1-tp).cumsum()[-1]
        tpc = tp.cumsum()[-1]

        recall = tpc / (n_l + eps)
        precision = tpc / (tpc + fpc)

        print('cumsum recall : {} , precision : {}'.format(recall, precision))


            

    
    print('speed : {}'.format((dt[1]/seen) * 1E3))    



    # for batch_i, (img, im0, targets, paths, shapes, img_id) in enumerate(pbar) :
        
    #     t1 = time_sync()
    #     #img = img.to(device) #numpy array 
    #     img /= 255
    #     t2 = time_sync()

    #     # 

    #     dt[0] += t2 - t1
    #     # Actually run inference
        
    #     out = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        
    #     dt[1] += time_sync() - t2

    #     t3 = time_sync()

    #     out = out.cpu().detach().numpy()[-1].reshape((1, -1, nc + 5))
    #     out = non_max_suppression_np(out, args.conf_thres, args.iou_thres, args.classes, args.agnostic_nms)
    #     dt[2] += time_sync() - t3

    #     seen += 1

if __name__ == '__main__':
    main()