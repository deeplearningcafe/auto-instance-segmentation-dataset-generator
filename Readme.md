Pipeline:
1. Get screnshots ----> 2. Filter by similarity to get unique images. ------> 3. Predict bbox with detector(yolox). ------> 4. Revise the annotations(cvat). ------> 5. Use SAM to get the masks.
Result: A instance segmentation dataset.

We won't include training scripts.

Detection Pipeline:
1. Make Predictions ------> 2. Transform predictions to annotations in supervision ----> 3. Export to COCO format. -----> 4. Update to cvat format.
The annotation file path should be predeterminated, just given a folder name, inside include images and annotations.json


# Install
torch @ https://github.com/pytorch/pytorch/archive/refs/tags/v2.2.1.zip

## MMDetection
https://mmdetection.readthedocs.io/en/latest/get_started.html
Install MMEngine and MMCV using MIM
```bash
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```
```
├── data
│   ├── inputs
│   │   ├── video-file
│   ├── outputs
│   │   ├── 
│   │   ├── video-file
│   │   │   ├── screenshots
│   │   │   ├── uniques
│   │   │   ├── video-file-dataset
│   │   │   │   ├── images
│   │   │   │   ├── annotations
├── pretrained_models
├── src
│   ├── characterdataset
│   │   ├── common
│   │   ├── configs
│   │   ├── datasetmanager
│   │   ├── oshifinder
│   │   ├── train_llm
├── tests
│   ├── test_dataset_manager.py
│   ├── test_finder.py
│   ├── test_train_conversational.py
├── webui_finder.py
├── train_webui.py
```


