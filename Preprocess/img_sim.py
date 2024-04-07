import torch
from transformers import AutoProcessor, CLIPVisionModelWithProjection, CLIPProcessor
from PIL import Image
import os
from torch.nn.functional import cosine_similarity
import shutil
from tqdm import tqdm
from transformers.trainer_utils import set_seed
import json
import gc

from utils.logs import create_logger

log = create_logger(__name__)

set_seed(1234)

def copy_file(input_path:str=None, output_path:str=None, index:str=None, text:str=None) -> str:
    """Copies a given file to a given location and returns the new file path

    Args:
        input_path (str, optional): path of the file. Defaults to None.
        output_path (str, optional): directory to copy the file. Defaults to None.
        index (str, optional): index for the name. Defaults to None.
        text (str, optional): text for the name. Defaults to None.
    Returns:
        str: new file path
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # update the names to include text, to add audios from other annotations
    new_filename = text
    # ファイルをコピーして、名前を変更する
    shutil.copy2(input_path, os.path.join(output_path, new_filename))
    
    copied_file = os.path.join(output_path, new_filename)
    return copied_file

def get_image_embeddings(image_paths: list[str], clip_model: CLIPVisionModelWithProjection, clip_processor: CLIPProcessor, batch_size=128) -> torch.Tensor:
    """ Given a list of image paths, opens them and uses clip for extracting the embeddings. Returns those embeddings

    Args:
        image_paths (list[str]): list of images paths
        clip_model (CLIPVisionModelWithProjection): clip model
        clip_processor (CLIPProcessor): clip processor
        batch_size (int, optional): batch size for extraction of embeddings. Defaults to 128.

    Returns:
        torch.Tensor: the embeddings tensor
    """
    embeddings = []
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(range(num_batches), "creating embeddings"):
            torch.cuda.empty_cache()
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]
            batch_images = [Image.open(img_path).convert("RGB") for img_path in batch_paths]
            inputs = clip_processor(images=batch_images, return_tensors="pt").to("cuda")
            outputs = clip_model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(image_embeds.to("cpu"))
    return torch.cat(embeddings)


def remove_similar_images(image_folder:str, similarity_threshold:float=0.95, output_path:str="output_path") -> None:
    """Given a image folder, it selects images that are unique based on the embeddings similarity. Then saves them in the output_path.
    The algorithm is as follows, loops all the embeddings from the get_image_embeddings, then compares the actual embedding with the
    already saved embeddings, if similarity below threshold then add that image to the unique folder and add its embedding to the 
    already saved embeddings list. This ensures that from similar images we save one of them. As we save the embeddings in a 
    new list, there is no need to recompute them, so we only compute all the embeddings once.

    Args:
        image_folder (str): path of the image where the images to select are stored
        similarity_threshold (float, optional): similarity threshold, if similarity higher than the th then not unique. Defaults to 0.95.
        output_path (str, optional): _description_. Defaults to "output_path".
    """
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    embeddings = get_image_embeddings(image_paths, clip_model, clip_processor)

    unique_images = []
    # the first image is always added to be able to start comparing
    image_name = os.path.basename(image_paths[0])
    copy_file(image_paths[0], output_path, index=str(0), text=image_name)
    
    # get the embeddings of the saved files, if there is already a similar image in the saved images, then skip
    embeddings_saved = [embeddings[0]]

    # loop all the embeddings list
    for i in tqdm(range(1, len(image_paths)), "Looping the image folder"):
        is_unique = True
        
        # compare the actual embedding with all the already saved embeddings
        for j in range(len(embeddings_saved)):
            similarity_score = cosine_similarity(embeddings[i].unsqueeze(0), embeddings_saved[j].unsqueeze(0))
            # if similarity score high that means that in the unique folder there is already a similar image so skip
            if similarity_score > similarity_threshold:
                is_unique = False
                continue
            
        if is_unique:
            # copy to the saved folder
            image_name = os.path.basename(image_paths[i])
            copy_file(image_paths[i], output_path, index=i, text=image_name)
            embeddings_saved.append(embeddings[i])
        
    # delete the model and release memory
    # model will still be on cache until its place is taken by other objects so also execute the below lines
    del clip_model
    gc.collect()
    torch.cuda.empty_cache()

