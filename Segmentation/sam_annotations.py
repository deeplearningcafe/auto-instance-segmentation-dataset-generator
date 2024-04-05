import numpy as np
from segment_anything import SamPredictor
import json
import supervision as sv
import cv2
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
import torch
from datetime import datetime
from tqdm import tqdm
import logging
import gc
import os

from utils.logs import create_logger

log = create_logger(__name__)

def segment(sam_predictor: SamPredictor, image: np.ndarray, 
            xyxy: np.ndarray) -> np.ndarray:
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
    x_f = bbox[0] + bbox[2]
    y_f = bbox[1] + bbox[3] 
    bbox[2:] = [x_f ,y_f]
    
    return bbox

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def load_img(img_path:str=None) -> np.ndarray:
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def mask_2_poly(mask: np.ndarray):
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
    

# sv.mask_to_polygons(mask: np.ndarray) -> List[np.ndarray]

# with open('test/annotations/instances_default.json', 'r') as file:
#     json_load = json.load(file)
# file.close()
# print(json_load.keys())

# # segmentation = [[1689.27,121.41,1653.89,133.9,1639.31,130.78,1619.54,146.39,1593.52,126.62,1579.47,126.1,1555.53,146.91,1520.14,202.08,1488.91,287.95,1482.67,321.77,1495.68,358.72,1500.36,387.35,1488.39,410.24,1476.94,414.93,1449.88,399.84,1456.65,418.05,1475.38,431.06,1500.88,425.86,1485.27,451.88,1474.34,492.99,1477.46,518.49,1487.87,538.27,1500.36,534.62,1504.53,549.72,1502.96,567.93,1452.48,586.15,1437.91,597.07,1454.05,618.41,1452.48,640.27,1463.41,665.77,1427.5,711.05,1424.9,719.89,1405.13,728.74,1416.58,746.43,1427.5,762.57,1431.67,785.99,1456.13,800.56,1464.97,805.76,1436.35,922.86,1451.96,915.05,1460.29,911.41,1491.52,921.81,1519.62,924.42,1528.46,857.8,1544.08,924.94,1562.29,1004.04,1543.56,1052.96,1535.23,1077.94,1599.5,1080.0,1616.58,1080.0,1639.31,1005.6,1630.47,867.17,1648.16,869.77,1667.42,998.84,1678.35,1036.31,1684.07,1077.94,1753.81,1077.42,1741.84,1037.87,1739.23,1020.17,1743.92,945.23,1745.48,898.4,1772.02,884.87,1765.25,845.83,1782.95,825.02,1811.57,816.17,1818.86,790.67,1847.48,763.09,1825.1,751.64,1781.39,678.78,1770.46,670.45,1783.47,644.95,1777.22,636.63,1804.29,627.26,1789.71,607.48,1773.06,598.12,1771.5,538.27,1781.39,532.02,1784.51,515.37,1792.32,491.95,1773.06,458.12,1762.65,445.11,1767.34,422.73,1768.38,407.12,1757.45,373.82,1743.92,380.06,1747.56,363.41,1738.71,357.68,1782.43,316.57,1778.79,289.51,1761.09,251.52,1763.69,230.7,1755.37,221.85,1754.85,215.61,1767.34,214.05,1786.59,226.02,1789.71,210.4,1772.02,185.42,1757.97,171.89,1738.71,182.82,1736.11,180.22,1741.84,168.77,1722.58,141.19,1717.38,139.11,1709.57,146.39,1705.41,145.35,1706.45,138.07]]


# seg = lambda lst: [[np.int32(lst[i]), np.int32(lst[i+1])] for i in range(0, len(lst)-1, 2)]
# flatten = lambda lst: [item for sublist in lst for item in sublist]
# # input_list = [[1689.27, 121.41], [1653.89, 133.9], [1639.31, 130.78], [1619.54, 146.39], [1593.52]]

# """output_list = flatten(input_list)
# print(output_list)


# output = seg(segmentation[0])
# print(output)
# # s = (np.array(seg).reshape(-1, 2) / np.array([1980, 1080])).reshape(-1).tolist()
# # print(s)

# mask = np.zeros((1080, 1980))

# cv2.fillPoly(mask, np.array([output], dtype=np.int32), color=1)


# plt.figure(figsize=(20,20))
# plt.imshow(np.zeros((1080, 1980)))
# # show_anns(masks)
# show_mask(mask, plt.gca())
# plt.axis('off')
# # plt.savefig("outputs/sam/segmentation_sam_bbox.png")
# plt.show()

# # masks = [ sv.polygon_to_mask(p,(1980, 1080)) for p in output ]
# print(mask.shape, sum(mask))
# print(sv.mask_to_polygons(mask))
# print(len(sv.mask_to_polygons(mask)), sv.mask_to_polygons(mask)[0].shape)"""


# for image in range(len(json_load["images"])):
#     print(json_load["images"][image])
#     break

# print(json_load["annotations"][0].keys())



# """
# So SAM outputs binary masks, but we need polygon shapes, which is the output of the cvat annotations I 
# made, so we need to convert  those binary masks to polygons.

# "iscrowd": 0 if your segmentation based on polygon (object instance)
# "iscrowd": 1 if your segmentation based uncompressed RLE (crowd) this is using the mask tool of cvat

# We need to loop all the images, then loop the annotations to look for the image id
# then from those annotations use the bbox to get the binary mask and then
# change the binary mask to polygon and append.
# As the annotations list is bigger than the img list, 
# we will loop the annotations and then for each img in annotations loop the img list.
# """

# model_name = "vit_l"
# sam = sam_model_registry[model_name](checkpoint="/home/victor/projects/mmdetection/checkpoints/sam_vit_l_0b3195.pth")

# # モデルをGPUに載せる
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# sam.to(device=DEVICE)
# predictor = SamPredictor(sam)

# img_base_path = "test/images"
# start = datetime.now()

# updated_samples = 0
# for i in tqdm(range(len(json_load["annotations"]))):
#     torch.cuda.empty_cache()
    
#     # check if already has segmentation
#     if len(json_load["annotations"][i]["segmentation"]) > 0:
#         log.info(f"Skipping index {i}")
#         continue
    
#     img_id = json_load["annotations"][i]["image_id"]
#     for j in range(len(json_load["images"])):
#         if img_id == json_load["images"][j]["id"]:
#             img_name = json_load["images"][j]["file_name"]
#             bbox = convert_wh_2_xy(json_load["annotations"][i]["bbox"])
#             bbox = np.array(bbox)
#             image = load_img(f"{img_base_path}/{img_name}")
#             mask = segment(predictor, image, bbox[None, :])
#             # now we have a binary mask so we need to get the polygons, it outputs a list
#             # polygons = sv.mask_to_polygons(mask[0][0])
#             polygons = mask_2_poly(mask[0][0])
            
            
#             # this creates a list for each polygon so one mask has several polygons
#             # [[175, 758, 174, 757, 174, 756, 173, 755, 173, 750], [324, 708,  325, 710, 324, 709], [205, 689, 200, 697, 200, 694]]
            

#             # this is of the shape [[x, y], [x, y], [x, y]] but we want [x1, y1, x2, y2, x3, y3]
#             # segmentation = flatten(polygons)
#             # print(segmentation)
#             # update the segmentation field of the annotations
#             json_load["annotations"][i]["segmentation"] = polygons
#             # print(json_load["annotations"][i]["segmentation"])

#             updated_samples += 1
#             # once it has updated this annotation, go to the next one
#             continue

# log.info(f"Peak memory: {torch.cuda.max_memory_allocated()*1e-9}")
# log.info(f"Time to complete: {datetime.now()-start}")
# log.info(f"Updated samples {updated_samples}")
# filename =  "test/annotations/instances_default_updated.json"
# with open(filename, "w") as jsonFile:
#     json.dump(json_load, jsonFile)
# log.info(f"File saved with name {filename}")


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def segment_dataset_sam(json_path:str=None, base_path:str=None) -> None:
    
    with open(json_path, 'r') as file:
        json_load = json.load(file)
    file.close()
    log.info(json_load.keys())

    model_name = "vit_l"
    sam = sam_model_registry[model_name](checkpoint="/home/victor/projects/mmdetection/checkpoints/sam_vit_l_0b3195.pth")

    # モデルをGPUに載せる
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    start = datetime.now()

    updated_samples = 0


    for i in tqdm(range(len(json_load["annotations"]))):
        torch.cuda.empty_cache()
        
        # check if already has segmentation
        if len(json_load["annotations"][i]["segmentation"]) > 0:
            log.info(f"Skipping index {i}")
            continue
        
        img_id = json_load["annotations"][i]["image_id"]
        for j in range(len(json_load["images"])):
            if img_id == json_load["images"][j]["id"]:
                img_name = json_load["images"][j]["file_name"]
                bbox = convert_wh_2_xy(json_load["annotations"][i]["bbox"])
                bbox = np.array(bbox)
                image = load_img(f"{base_path}/{img_name}")
                mask = segment(predictor, image, bbox[None, :])
                # now we have a binary mask so we need to get the polygons, it outputs a list
                # polygons = sv.mask_to_polygons(mask[0][0])
                polygons = mask_2_poly(mask[0][0])
                
                
                # this creates a list for each polygon so one mask has several polygons
                # [[175, 758, 174, 757, 174, 756, 173, 755, 173, 750], [324, 708,  325, 710, 324, 709], [205, 689, 200, 697, 200, 694]]
                

                # this is of the shape [[x, y], [x, y], [x, y]] but we want [x1, y1, x2, y2, x3, y3]
                # segmentation = flatten(polygons)
                # print(segmentation)
                # update the segmentation field of the annotations
                json_load["annotations"][i]["segmentation"] = polygons
                # print(json_load["annotations"][i]["segmentation"])

                updated_samples += 1
                # once it has updated this annotation, go to the next one
                continue

    log.info(f"Peak memory: {torch.cuda.max_memory_allocated()*1e-9}")
    log.info(f"Time to complete: {datetime.now()-start}")
    log.info(f"Updated samples {updated_samples}")
    anntations_file = os.path.basename(json_path)
    anntations_filename, format = os.path.splitext(anntations_file)
    filename =  f"{anntations_filename}_updated.json"
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