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
from mmdet.apis import DetInferencer
import torch
import supervision as sv
from mmdet.structures import DetDataSample
from tqdm import tqdm
import cv2
import os
import json
import gc

from utils.logs import create_logger

# logging.basicConfig(encoding='utf-8', level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

# log = logging.getLogger(__name__)
log = create_logger(__name__)

class YoloXDetector:
    """ Yolox Detector class"""
    def __init__(self, config_file:str, checkpoint_file:str) -> None:
        self.model = self.init_model(config_file, checkpoint_file)
        
    
    def init_model(self, config_file:str, checkpoint_file:str) -> DetInferencer:
        """Creates the model instance

        Args:
            config_file (str): the model configuration of mmdetection 
            checkpoint_file (str): the checkpoint of the model

        Returns:
            DetInferencer: inference class
        """
        device = 'cuda:0'
        inferencer = DetInferencer(model=config_file, weights=checkpoint_file, device=device)
        return inferencer

    def predict(self, img_dir:str, batch_size:int=64,) -> DetDataSample:
        """ Given a directory of images, predicts the bboxes using batched samples.

        Args:
            img_dir (str): directory of images to predict
            batch_size (int, optional): batch size for inference. Defaults to 64.

        Returns:
            DetDataSample: object containing all the annotations
        """
        img_files = os.listdir(img_dir)
        # we need to add the whole path to each img
        img_files = [os.path.join(img_dir, img) for img in img_files]
        log.info(len(img_files))

        predictions =  self.model(img_files, no_save_pred=True, return_datasamples=True, 
                                  print_result=False, batch_size=batch_size,)
    
        return predictions

class PredictionProcessor:
    """ 
    Class for processing the predictions from Detector and creates a COCO dataset
    """
    MIN_IMAGE_AREA_PERCENTAGE = 0.002
    MAX_IMAGE_AREA_PERCENTAGE = 0.90
    APPROXIMATION_PERCENTAGE = 0.75

    def __init__(self, classes:list[str], annotations_folder:str) -> None:        
        self.classes = classes
        self.images_directory = os.path.join(annotations_folder, "images")
        self.annotations_file = os.path.join(annotations_folder, "annotations")
        self.annotations_file = os.path.join(self.annotations_file, "annotations.json")
        
    
    @staticmethod
    def process_predictions(predictions:list[DetDataSample], conf_threshold:float=0.2) -> tuple[list[sv.Detections], list[str]]:
        """Converts the predictions to supervision format and creates a dictionary with the images objects

        Args:
            predictions (list[DetDataSample]): output from the detector
            conf_threshold (float, optional): minimum confidence of the detector to use the prediction. Defaults to 0.2.

        Returns:
            tuple[list[sv.Detections], list[str]]: annotations and images dictionaries.
        """
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

    def format_cvat(self, json_path:str) -> None:
        """Cvat does not allow ids to start by 0 so add 1 to the ids.

        Args:
            json_path (str): annotations file from COCO dataset.
        """
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

    def export_dataset_coco(self, images:dict, annotations:dict) -> tuple[str, str]:
        """ Exports the annotations and images from process_predictions to COCO format.

        Args:
            images (dict): images dict 
            annotations (dict): supervision annotations 

        Returns:
            tuple[str, str]: annotations file and images directory
        """
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

        images[pred.img_path] = cv2.imread(pred.img_path)
        detections = sv.Detections.from_mmdetection(pred)
        detections = detections[keep_idx]

        annotations[pred.img_path] = detections
        
        
    return annotations, images



def create_dataset(conf) -> tuple[str, str]:
    """Given the configuration with the images folder, it creates a COCO dataset
    
    Args:
        conf (_type_): configuration object

    Returns:
        tuple[str, str]: annotations file and images directory
    """
    # Load detector and post_processor
    detector = YoloXDetector(conf.detector.config_file, conf.detector.checkpoint_file)
    folder_name = os.path.basename(conf.folder_path)
    print(conf.folder_path)
    annotations_folder = os.path.join(conf.folder_path, f"dataset")
    print(annotations_folder)
    processor = PredictionProcessor(classes=conf.detector.classes, annotations_folder=annotations_folder)
    
    # 1. Make Predictions
    predictions = detector.predict(conf.uniques_folder, batch_size=64)

    # 2. Transform predictions to annotations in supervision
    annotations, images = processor.process_predictions(predictions, conf_threshold=conf.detector.conf_threshold)
    
    # 3. Export to COCO format and cvat format
    annotations_file, images_directory = processor.export_dataset_coco(images, annotations)
    
    # delete the model and release memory
    # model will still be on cache until its place is taken by other objects so also execute the below lines
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    
    return annotations_file, images_directory

