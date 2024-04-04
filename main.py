import os

from Preprocess.screenshots import get_frames
from Preprocess.img_sim import remove_similar_images
from Detector.yolox_annotations import create_dataset
from Segmentation.sam_annotations import segment_dataset_sam
from utils.config import load_global_config
from utils.logs import create_logger

log = create_logger(__name__)

def main(conf):
    # 1. Get screnshots
    log.info("Get screenshots")
    if not os.path.isfile(conf.video_path):
        raise ValueError(
                f"The annotation file at {conf.video_path} does not exists"
            )
    file = os.path.basename(conf.video_path)
    filename, format = os.path.splitext(file)
    conf.output_path = os.path.join(current_dir, conf.output_path)
    conf.folder_path = os.path.join(conf.output_path, filename)
    # here output the screenshots
    screenshots_folder = os.path.join(conf.folder_path, conf.screenshots_folder)
    log.info(screenshots_folder)
    os.makedirs(screenshots_folder, exist_ok=True)
    get_frames(conf.video_path, screenshots_folder, time_lapse=24, 
               video_name=filename)
    
    # 2. Filter similarity
    # if not os.path.exists(conf.image_folder):
    #     raise ValueError(
    #         f"The annotation file at {conf.image_folder} does not exists"
    #     )
    # images should be outputed to the output_path directory
    # output_path_unique = os.path.join(OUTPUT_PATH, OUTPUT_UNIQUE)
    # we can directly update the config instead of creating new variables
    conf.uniques_folder = os.path.join(conf.folder_path, conf.uniques_folder)
    log.info(conf.uniques_folder)
    try:  
        # creating a folder named data 
        if not os.path.exists(conf.uniques_folder): 
            os.makedirs(conf.uniques_folder)
    except OSError:
        log.info ('Error! Could not create a directory') 

    remove_similar_images(screenshots_folder, similarity_threshold=conf.similarity_threshold, output_path=conf.uniques_folder)
    
    # 3. Make annotations with Detector
    # First we need to make the predictions
    annotations_file, images_directory = create_dataset(conf)
    
    # (Optional) 4. Revise the annotations(cvat).
    if not conf.revise_annotations:
        # 5. Use SAM to get the masks.
        segment_dataset_sam(annotations_file, images_directory)
    

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    conf_path = os.path.join(current_dir, "default_config.toml")

    conf = load_global_config(conf_path)
    main(conf)