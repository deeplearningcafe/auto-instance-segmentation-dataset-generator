import torch
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from PIL import Image
import os
from torch.nn.functional import cosine_similarity
import shutil
from tqdm import tqdm
from transformers.trainer_utils import set_seed
import json

set_seed(1234)


# clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
# clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

# img_path=r"asuka dibujo.jpg"
# with torch.no_grad():
#     inputs = clip_processor(images=Image.open(img_path), return_tensors="pt")
#     outputs = clip_model(**inputs)
#     image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)

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

def get_image_embeddings(image_paths, clip_model, clip_processor, batch_size=128):
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


def remove_similar_images(image_folder, similarity_threshold=0.95, output_path:str="output_path"):
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    embeddings = get_image_embeddings(image_paths, clip_model, clip_processor)

    unique_images = []
    image_name = os.path.basename(image_paths[0])
    copy_file(image_paths[0], output_path, index=str(0), text=image_name)
    
    # get the embeddings of the saved files, if there is already a similar image in the saved images, then skip
    # image_paths_saved = [os.path.join(output_path, img) for img in os.listdir(output_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    # embeddings_saved = get_image_embeddings(image_paths_saved, clip_model, clip_processor)
    embeddings_saved = [embeddings[0]]
    # instead of computing the embeddings, we just compute them one time and add to this list

    for i in tqdm(range(1, len(image_paths)), "Looping the image folder"):
        is_unique = True
        
        for j in range(len(embeddings_saved)):
            similarity_score = cosine_similarity(embeddings[i].unsqueeze(0), embeddings_saved[j].unsqueeze(0))
            if similarity_score > similarity_threshold:
                # print(similarity_score)
                is_unique = False
                continue
            
        if is_unique:
            # copy to the saved folder
            image_name = os.path.basename(image_paths[i])
            copy_file(image_paths[i], output_path, index=i, text=image_name)
            embeddings_saved.append(embeddings[i])
        



output_path = "unique_images"
try:  
    # creating a folder named data 
    if not os.path.exists(output_path): 
        os.makedirs(output_path)

except OSError:
    print ('Error! Could not create a directory') 

image_folder = "e:/Data/object_dection/maids"
remove_similar_images(image_folder, output_path=output_path)
print(len(os.listdir(output_path)))
