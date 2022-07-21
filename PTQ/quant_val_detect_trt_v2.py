import os
import ctypes
import time
import sys
import argparse

import cv2
import numpy as np
from PIL import Image
import tensorrt as trt
<<<<<<< HEAD
from utils.general import non_max_suppression_np

=======
from PTQ.utils.general import non_max_suppression
>>>>>>> 851fcd4c2ccdaa9255236d431399fc4b32404c97

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

# COCO label list
COCO_LABELS = coco_utils.COCO_CLASSES_LIST

# Model used for inference
<<<<<<< HEAD
MODEL_NAME = 'yolov5l'
=======
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
>>>>>>> 851fcd4c2ccdaa9255236d431399fc4b32404c97

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
    parser.add_argument('-p', '--precision', type=int, choices=[32, 16, 8], default=32,
        help='desired TensorRT float precision to build an engine with')
    parser.add_argument('-b', '--max_batch_size', type=int, default=1,
        help='max TensorRT engine batch size')
    parser.add_argument('-w', '--workspace_dir',default='PTQ',
        help='sample workspace directory')
    parser.add_argument('-fc', '--flatten_concat',
        help='path of built FlattenConcat plugin')
    parser.add_argument('-d', '--calib_dataset', default='/home/youngjin/datasets/coco/val/images',
        help='path to the calibration dataset')
    parser.add_argument('--imgsz', default=(640,640))
    parser.add_argument('--workers', default=8)
<<<<<<< HEAD
    parser.add_argument('--device', default='0')
=======
    parser.add_argument('--device', default='0,1')
>>>>>>> 851fcd4c2ccdaa9255236d431399fc4b32404c97
    parser.add_argument('--resize', default=True)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--stride', default=32)
    parser.add_argument('--conf_thres', default=0.45)
    parser.add_argument('--iou_thres', default=0.25)
    parser.add_argument('--data', default='../dataset/coco.yaml')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic_nms', default=False)
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

    # Fetch .uff model path, convert from .pb
    # if needed, using prepare_ssd_model
    yolo_model_onnx_path = PATHS.get_model_onnx_path(MODEL_NAME)
    if not os.path.exists(yolo_model_onnx_path):
        model_utils.prepare_yolo_model(MODEL_NAME)

    # Set up all TensorRT data structures needed for inference
    trt_inference_wrapper = inference_utils.TRTInference(
        args.trt_engine_path, yolo_model_onnx_path,
        trt_engine_datatype=args.trt_engine_datatype,
        calib_dataset = args.calib_dataset,
        batch_size=args.max_batch_size)

    print("TRT ENGINE PATH", args.trt_engine_path)


    dataset = create_dataloader_custom(image_path=args.input_img_path+'/images',
                                                    label_path=args.input_img_path+'/labels',
                                                    imgsz=args.imgsz,
                                                    batch_size=args.batch_size,
                                                    stride=args.stride,
                                                    workers=args.workers, 
                                                    resize=args.resize)

    with open(data, error='ignore') as f :
        data = yaml.safe_load(f)
    nc = int(data['nc'])
    seen = 0
    pbar = tqdm(dataset)

    dt = [0.0, 0.0, 0.0]

    for batch_i, (img, im0, targets, paths, shapes, img_id) in enumerate(pbar) :
        
        t1 = time_sync()
        img = img.to(device)
        img /= 255
        t2 = time_sync()
        dt[0] += t2 - t1
        # Actually run inference
        out, keep_count_out = trt_inference_wrapper.infer(img)

        dt[1] += time_sync() - t2

        t3 = time_sync()

        out = out.cpu().detach().numpy()[-1].reshape((1, -1, nc + 5))
        out = non_max_suppression_np(out, args.conf_thres, args.iou_thres, args.classes, args.agnostic_nms)
        dt[2] += time_sync() - t3

        seen += 1

        # Overlay the bounding boxes on the image
        # let analyze_prediction() draw them based on model output
        img_pil = Image.fromarray(img)
        prediction_fields = len(TRT_PREDICTION_LAYOUT)
        for det in range(int(keep_count_out[0])):
            analyze_prediction(out, det * prediction_fields, img_pil)
        final_img = np.asarray(img_pil)


if __name__ == '__main__':
    main()