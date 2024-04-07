import numpy as np
from segment_anything import SamPredictor
import json
import cv2
from segment_anything import SamPredictor, sam_model_registry
import torch
from datetime import datetime
from tqdm import tqdm
import gc
import os

from utils.logs import create_logger

log = create_logger(__name__)

def segment(sam_predictor: SamPredictor, image: np.ndarray, 
            xyxy: np.ndarray) -> np.ndarray:
    """ Predicts masks objects given an image and a bbox using SAM.

    Args:
        sam_predictor (SamPredictor): sam model to predict
        image (np.ndarray): image
        xyxy (np.ndarray): array with bboxes

    Returns:
        np.ndarray: an array of masks
    """
    sam_predictor.set_image(image)
    result_masks = []
    
    for box in xyxy:
        mask, scores, logits = sam_predictor.predict(
            box=box,
            point_coords=None,
            point_labels=None,
            multimask_output=False
        )
        # index = np.argmax(scores)
        result_masks.append(mask)
    
    return np.array(result_masks)

def convert_wh_2_xy(bbox:list=None) -> list[float]:
    """Converts from COCO format (x_o, y_o, w, h) to (x_o, y_o, x_f, y_f)

    Args:
        bbox (list, optional): bbox to format. Defaults to None.

    Returns:
        list[float]: formatted bbox
    """
    x_f = bbox[0] + bbox[2]
    y_f = bbox[1] + bbox[3] 
    bbox[2:] = [x_f ,y_f]
    
    return bbox


def load_img(img_path:str=None) -> np.ndarray:
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def mask_2_poly(mask: np.ndarray):
    """Converts segmentation masks to polygons

    Args:
        mask (np.ndarray): array with the masks

    Returns:
        _type_: list of polygons
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    segmentation = []

    for contour in contours:
        contour = contour.flatten().tolist()
        # segmentation.append(contour)
        if len(contour) > 4:
            segmentation.append(contour)
    return segmentation
    


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def segment_dataset_sam(json_path:str=None, base_path:str=None) -> str:
    """ Given a coco dataset, image folder and annotations file, it creates segmentations masks and adds them
    to the annotation file.

    Args:
        json_path (str, optional): path to the annotations file with the bbox. Defaults to None.
        base_path (str, optional): path to the folder where the images of the annotation file 
            are stored. Defaults to None.

    Returns:
        str: new annotation file with updated segmentation masks
    """
    # open annotations
    with open(json_path, 'r') as file:
        json_load = json.load(file)
    file.close()
    log.info(json_load.keys())

    # create sam instance
    model_name = "vit_l"
    sam = sam_model_registry[model_name](checkpoint="/home/victor/projects/mmdetection/checkpoints/sam_vit_l_0b3195.pth")

    # モデルをGPUに載せる
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    start = datetime.now()

    updated_samples = 0

    # loop all the annotations, as usually number higher than images is better this way
    for i in tqdm(range(len(json_load["annotations"]))):
        torch.cuda.empty_cache()
        
        # check if already has segmentation
        if len(json_load["annotations"][i]["segmentation"]) > 0:
            log.info(f"Skipping index {i}")
            continue
        
        # loop all images looking for the image of the actual annotation
        img_id = json_load["annotations"][i]["image_id"]
        for j in range(len(json_load["images"])):
            if img_id == json_load["images"][j]["id"]:
                img_name = json_load["images"][j]["file_name"]
                
                # format bbox
                bbox = convert_wh_2_xy(json_load["annotations"][i]["bbox"])
                bbox = np.array(bbox)
                image = load_img(f"{base_path}/{img_name}")
                
                # predict with sam
                mask = segment(predictor, image, bbox[None, :])
                # now we have a binary mask so we need to get the polygons, it outputs a list
                polygons = mask_2_poly(mask[0][0])
                
                # this creates a list for each polygon so one mask has several polygons
                # [[175, 758, 174, 757, 174, 756, 173, 755, 173, 750], [324, 708,  325, 710, 324, 709], [205, 689, 200, 697, 200, 694]]
                
                # update the segmentation field of the annotations
                json_load["annotations"][i]["segmentation"] = polygons

                updated_samples += 1
                # once it has updated this annotation, go to the next one
                continue

    log.info(f"Peak memory: {torch.cuda.max_memory_allocated()*1e-9}")
    log.info(f"Time to complete: {datetime.now()-start}")
    log.info(f"Updated samples {updated_samples}")
    
    # save the annotations file
    annotations_file = os.path.basename(json_path)
    annotations_folder = os.path.dirname(json_path)
    annotations_filename, format = os.path.splitext(annotations_file)
    filename =  f"{annotations_folder}/{annotations_filename}_updated.json"
    with open(filename, "w") as jsonFile:
        json.dump(json_load, jsonFile)
    log.info(f"File saved with name {filename}")
    
    del sam
    gc.collect()
    torch.cuda.empty_cache()
    
    return filename

if __name__ == "__main__":
    json_path = ""
    base_path = ""
    
    segment_dataset_sam(json_path, base_path)