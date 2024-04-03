"""
We store the images in a folder, then we loop that image folder, as we want to batch them, we can list dir then use batch as range for the loop.
The difficult part is to create the annotation file.
We need only the image name in the annotations, the whole path of the image is not needed. Also we need to change the xyx2y2 to cxcywh format.


With this we can create coco dataset, we just need to create the Detections objects, and create a list with the images path.
We don't need to filter the images manually, just use the thresholds to decide. But there might still be really similar images, we can measure the image similarity and 
if really similar delete them. This should be done before predicting, prediction is the process that needs more time and resources so if possible, just want to predict 
good images.

Batch 2: 238'519'808, Batch 5: 338'770'432  Memory per sample: 33'416'874.666666668 so each sample is like 300 MB so we can until batch size of 14 maybe

https://github.com/roboflow/supervision/blob/develop/supervision/dataset/formats/coco.py#L100
"""
from mmdet.apis import init_detector, inference_detector, DetInferencer
import pathlib
import torch
import supervision as sv
from mmdet.structures import DetDataSample
from tqdm import tqdm
import cv2
import os
import json
import logging

logging.basicConfig(encoding='utf-8', level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

log = logging.getLogger(__name__)

class YoloXDetector:
    def __init__(self, config_file:str, checkpoint_file:str) -> None:
        self.model = self.init_model(config_file, checkpoint_file)
        
    
    def init_model(self, config_file:str, checkpoint_file:str) -> DetInferencer:
        device = 'cuda:0'
        inferencer = DetInferencer(model=config_file, weights=checkpoint_file, device=device)
        return inferencer

    def predict(self, img_dir:str, batch_size:int=64,):
        img_files = os.listdir(img_dir)
        # we need to add the whole path to each img
        img_files = [os.path.join(img_dir, img) for img in img_files]
        log.info(len(img_files))

        predictions =  self.model(img_files, no_save_pred=True, return_datasamples=True, 
                                  print_result=False, batch_size=batch_size,)
    
        return predictions

class PredictionProcessor:
    MIN_IMAGE_AREA_PERCENTAGE = 0.002
    MAX_IMAGE_AREA_PERCENTAGE = 0.90
    APPROXIMATION_PERCENTAGE = 0.75

    def __init__(self, classes:list[str], annotations_folder:str) -> None:        
        self.classes = classes
        self.images_directory = os.path.join(annotations_folder, "images")
        # annotations_folder/annotations/annotations.json
        self.annotations_file = os.path.join(annotations_folder, "annotations")
        self.annotations_file = os.path.join(self.annotations_file, "annotations.json")
        
    
    @staticmethod
    def process_predictions(predictions:list[DetDataSample], conf_threshold:float=0.2) -> tuple[list[sv.Detections], list[str]]:

        images = {}
        annotations = {}
        for pred in tqdm(predictions['predictions'], 'processing the predictions'):
            results = []
            # this is looping all the predictions so one pred is one image
            keep_idx = []
            for i, score in enumerate(pred.pred_instances.scores):
                # this is looping the list of scores so one score is one bbox
                if score > conf_threshold:
                    keep_idx.append(i)
            if len(keep_idx) < 1:
                # if no scores bigger than the threshold, the delete
                continue


            images[pred.img_path] = cv2.imread(pred.img_path)
            detections = sv.Detections.from_mmdetection(pred)
            detections = detections[keep_idx]

            annotations[pred.img_path] = detections
            
            
        return annotations, images

    def format_cvat(json_path:str) -> None:
        with open(json_path, 'r') as file:
            json_load = json.load(file)
        file.close()
        log.info(json_load.keys())
        
        for i in tqdm(range(len(json_load["categories"]))):
            json_load["categories"][i]["id"] += 1
        
        for i in tqdm(range(len(json_load["annotations"]))):
            json_load["annotations"][i]["category_id"] += 1

        with open(json_path, "w") as jsonFile:
            json.dump(json_load, jsonFile)
        log.info(f"File saved with name {json_path}")

    def export_dataset_coco(self, images:str, annotations:str):
        dataset = sv.DetectionDataset(
            classes=self.classes,
            images=images,
            annotations=annotations
        ).as_coco(
            images_directory_path=self.images_directory,
            annotations_path=self.annotations_file,
            min_image_area_percentage=PredictionProcessor.MIN_IMAGE_AREA_PERCENTAGE,
            max_image_area_percentage=PredictionProcessor.MAX_IMAGE_AREA_PERCENTAGE,
            approximation_percentage=PredictionProcessor.APPROXIMATION_PERCENTAGE
        )
        self.format_cvat(self.annotations_file)
        
        return self.annotations_file, self.images_directory
    
    



def process_predictions(predictions:list[DetDataSample], conf_threshold:float=0.2) -> tuple[list[sv.Detections], list[str]]:

    images = {}
    annotations = {}
    for pred in tqdm(predictions['predictions'], 'processing the predictions'):
        results = []
        # this is looping all the predictions so one pred is one image
        keep_idx = []
        for i, score in enumerate(pred.pred_instances.scores):
            # this is looping the list of scores so one score is one bbox
            if score > conf_threshold:
                keep_idx.append(i)
        if len(keep_idx) < 1:
            # if no scores bigger than the threshold, the delete
            continue
        # print(pred)
        # print(keep_idx)
        # print(pred.pred_instances.keys())
        # for key in pred.pred_instances.keys():
        #     print(key)
        #     print(pred.pred_instances[key], pred.pred_instances[key][keep_idx])
        #     pred.pred_instances[key] = pred.pred_instances[key][keep_idx]
        
        # images_paths.append(pred.img_path)
        
        # to create a dataset class using supervision we need to have dictionary with path and the image
        # results.append(sv.Detections.from_mmdetection(pred))

        images[pred.img_path] = cv2.imread(pred.img_path)
        detections = sv.Detections.from_mmdetection(pred)
        detections = detections[keep_idx]

        annotations[pred.img_path] = detections
        
        
    return annotations, images


def make_predictions(img_dir:str) -> dict[str, DetDataSample]:
    img_files = os.listdir(img_dir)
    # we need to add the whole path to each img
    img_files = [os.path.join(img_dir, img) for img in img_files]
    log.info(len(img_files))
    
    config_file = 'yolox_s_8x8_300e_maids.py'
    checkpoint_file = '../../work_dirs/yolox_s_8x8_300e_maids/epoch_64.pth'

    device = 'cuda:0'
    inferencer = DetInferencer(model=config_file, weights=checkpoint_file, device=device)

    predictions =  inferencer(img_files, no_save_pred=True, return_datasamples=True, print_result=False, batch_size=64,)
    
    return predictions

def format_cvat(json_path:str) -> None:
    with open(json_path, 'r') as file:
        json_load = json.load(file)
    file.close()
    log.info(json_load.keys())
    
    for i in tqdm(range(len(json_load["categories"]))):
        json_load["categories"][i]["id"] += 1
    
    for i in tqdm(range(len(json_load["annotations"]))):
        json_load["annotations"][i]["category_id"] += 1

    with open(json_path, "w") as jsonFile:
        json.dump(json_load, jsonFile)
    log.info(f"File saved with name {json_path}")

def create_dataset(conf):
    # Load detector and post_processor
    detector = YoloXDetector(conf.detector.config_file, conf.detector.checkpoint_file)
    folder_name = os.path.dirname(conf.folder_path)
    annotations_folder = os.path.join(conf.folder_path, f"{folder_name}_dataset")
    processor = PredictionProcessor(classes=conf.detector.classes, annotations_folder=annotations_folder)
    
    # 1. Make Predictions
    predictions = detector.predict(conf.uniques_folder, batch_size=64)

    # 2. Transform predictions to annotations in supervision
    annotations, images = processor.process_predictions(predictions, conf_threshold=conf.conf_threshold)
    
    # 3. Export to COCO format and cvat format
    annotations_file, images_directory = processor.export_dataset_coco(images, annotations)
    
    return annotations_file, images_directory

# images_path = r"/home/victor/projects/mmdetection/dataset/maids/yolox_images_for_ann/chapter5"
# predictions = make_predictions(images_path)
# log.info(round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2))
# log.info(round(torch.cuda.max_memory_reserved() / (1024 ** 3), 2))

# annotations, images = process_predictions(predictions=predictions, conf_threshold=0.65)

# log.info(len(images))
# metainfo = {
#     'classes': ('nagomi', 'ranko', 'yumechi', 'shiipon'),

# }
# IMAGES_DIRECTORY = f"yolox_annotations/chapter5/images"
# ANNOTATIONS_FILE = f"yolox_annotations/chapter5/annotations.json"


# MIN_IMAGE_AREA_PERCENTAGE = 0.002
# MAX_IMAGE_AREA_PERCENTAGE = 0.90
# APPROXIMATION_PERCENTAGE = 0.75

# updated the classes_to_coco_categories method in supervision/dataset/formats/coco.py to start counting from 1 so that label ids are from 1 to n
# we need to update the annotations I think it would be the best approach, because the annotations still start by 0
# dataset = sv.DetectionDataset(
#     classes=metainfo['classes'],
#     images=images,
#     annotations=annotations
# ).as_coco(
#     images_directory_path=IMAGES_DIRECTORY,
#     annotations_path=ANNOTATIONS_FILE,
#     min_image_area_percentage=MIN_IMAGE_AREA_PERCENTAGE,
#     max_image_area_percentage=MAX_IMAGE_AREA_PERCENTAGE,
#     approximation_percentage=APPROXIMATION_PERCENTAGE
# )

# format_cvat(ANNOTATIONS_FILE)