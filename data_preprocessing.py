import os
import json
import nltk
from nltk.tokenize import word_tokenize
from PIL import Image

# Set up paths
data_dir = "data"
coco_dir = os.path.join(data_dir, "coco")
nocaps_dir = os.path.join(data_dir, "nocaps")

# Load and process datasets
coco_annotation_file = os.path.join(coco_dir, "annotations", "captions_train2014.json")
nocaps_annotation_file = os.path.join(nocaps_dir, "annotations.json")

coco_data = []
nocaps_data = []

# Load COCO dataset
with open(coco_annotation_file, "r") as file:
    coco_annotations = json.load(file)["annotations"]

for annotation in coco_annotations:
    image_id = annotation["image_id"]
    caption = annotation["caption"]

    image_path = os.path.join(coco_dir, "train2014", f"COCO_train2014_{str(image_id).zfill(12)}.jpg")
    coco_data.append({"image_path": image_path, "caption": caption})

# Load NoCaps dataset
with open(nocaps_annotation_file, "r") as file:
    nocaps_annotations = json.load(file)["annotations"]

for annotation in nocaps_annotations:
    image_id = annotation["image_id"]
    caption = annotation["caption"]

    image_path = f"https://nocaps.org/images/{image_id}.jpg"
    nocaps_data.append({"image_path": image_path, "caption": caption})

dataset = coco_data + nocaps_data

# Preprocess captions
processed_data = []

for item in dataset:
    image_path = item["image_path"]
    caption = item["caption"]

    # Resize images (if necessary) using libraries like PIL
    image = Image.open(image_path)
    # Perform image resizing operations as needed

    # Tokenize captions using NLTK or Hugging Face Tokenizers
    tokenized_caption = word_tokenize(caption)

    # Add image path and tokenized caption to processed_data
    processed_data.append({
        "image_path": image_path,
        "caption": tokenized_caption
    })

# Save preprocessed data
preprocessed_data_path = os.path.join(data_dir, "preprocessed_data.json")

with open(preprocessed_data_path, "w") as file:
    json.dump(processed_data, file)