import numpy as np
import matplotlib.pyplot as plt
import json
import supervision as sv
import os

from Segmentation.sam_annotations import convert_wh_2_xy, load_img
from utils.logs import create_logger

log = create_logger(__name__)

# From https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def get_annotations(json_path:str=None, image_folder:str=None, save_plot:bool=False):
    # open annotations
    with open(json_path, 'r') as file:
        json_load = json.load(file)
    file.close()
    
    # loop all the annotations
    for i in range(len(json_load["annotations"])):
        # loop all images looking for the image of the actual annotation
        img_id = json_load["annotations"][i]["image_id"]
        for j in range(len(json_load["images"])):
            if img_id == json_load["images"][j]["id"]:
                img_name = json_load["images"][j]["file_name"]
                bbox = json_load["annotations"][i]["bbox"]
                bbox = np.array(bbox)
                image = load_img(f"{image_folder}/{img_name}")
                masks = json_load["annotations"][i]["segmentation"]
                
                # convert masks to [x1, y1], [x2, y2]
                
                seg = lambda lst: [[np.int32(lst[i]), np.int32(lst[i+1])] for i in range(0, len(lst)-1, 2)]
                masks = [seg(mask) for mask in masks]
                masks = [np.array(mask, dtype=np.int32) for mask in masks]
                log.info(f"Number of polygons {len(masks)}")
                
                masks = [sv.polygon_to_mask(p,(1980, 1080)) for p in masks]
                
                index = str(i).zfill(4)
                plot_sam(image, masks, bbox, save_plot, index)
            
            continue
        
        if i > 5:
            log.info("Stop showing plots")
            break
        
            

def plot_sam(image, masks, input_box, save_plot:bool=False, index:int=0):
    plt.figure(figsize=(10,5))
    plt.imshow(image)
    # plot all the masks, there are several polygons in the same annotation
    for mask in masks:
        show_mask(mask, plt.gca())
    show_box(input_box, plt.gca())
    plt.axis('off')
    if save_plot:
        outputs_path = "data/outputs/sam"
        os.makedirs(outputs_path, exist_ok=True)
        plt.savefig(f"{outputs_path}/segmentation_sam_bbox_{index}.png")
    plt.show()

if __name__ == "__main__":
    annotation_file = ""
    image_folder = ""
    get_annotations(annotation_file, image_folder)