import os
import requests
import json
from tqdm import tqdm
from PIL import Image

# Set up paths
data_dir = "data"
coco_dir = os.path.join(data_dir, "coco")
nocaps_dir = os.path.join(data_dir, "nocaps")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(coco_dir, exist_ok=True)
os.makedirs(nocaps_dir, exist_ok=True)

# Download COCO dataset
coco_images_url = "http://images.cocodataset.org/zips/train2014.zip"
coco_annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

coco_images_zip = os.path.join(coco_dir, "train2014.zip")
coco_annotations_zip = os.path.join(coco_dir, "annotations_trainval2014.zip")

print("Downloading COCO dataset...")
resp = requests.get(coco_images_url, stream=True)
with open(coco_images_zip, "wb") as file:
    for ck in tqdm(resp.iter_content(chunk_size=1024)):
        if ck:
            file.write(ck)

resp = requests.get(coco_annotations_url, stream=True)
with open(coco_annotations_zip, "wb") as file:
    for ck in tqdm(resp.iter_content(chunk_size=1024)):
        if ck:
            file.write(ck)

# Unzip COCO dataset
import zipfile

print("Unzipping COCO dataset...")
with zipfile.ZipFile(coco_images_zip, "r") as zip_ref:
    zip_ref.extractall(coco_dir)

with zipfile.ZipFile(coco_annotations_zip, "r") as zip_ref:
    zip_ref.extractall(coco_dir)

# Download NoCaps dataset
nocaps_url = "https://nocaps.org/data/standalone_annotations_v1.0.json"
nocaps_file = os.path.join(nocaps_dir, "annotations.json")

print("Downloading NoCaps dataset...")
resp = requests.get(nocaps_url)
with open(nocaps_file, "w") as file:
    file.write(resp.text)

# Load and process datasets
class COCODataset:
    def __init__(self, image_dir, annotation_file):
        self.image_dir = image_dir
        self.annotation_file = annotation_file

    def load_data(self):
        with open(self.annotation_file, "r") as file:
            annotations = json.load(file)["annotations"]

        data = []
        for annotation in annotations:
            image_id = annotation["image_id"]
            caption = annotation["caption"]

            image_path = os.path.join(self.image_dir, f"COCO_train2014_{str(image_id).zfill(12)}.jpg")
            image = Image.open(image_path).convert("RGB")

            data.append((image, caption))

        return data


class NoCapsDataset:
    def __init__(self, annotation_file):
        self.annotation_file = annotation_file

    def load_data(self):
        with open(self.annotation_file, "r") as file:
            annotations = json.load(file)["annotations"]

        data = []
        for annotation in annotations:
            image_id = annotation["image_id"]
            caption = annotation["caption"]

            image_path = f"https://nocaps.org/images/{image_id}.jpg"
            resp = requests.get(image_path)
            image = Image.open(BytesIO(resp.content)).convert("RGB")

            data.append((image, caption))

        return data

coco_image_dir = os.path.join(coco_dir, "train2014")
coco_annotation_file = os.path.join(coco_dir, "annotations", "captions_train2014.json")
coco_dataset = COCODataset(coco_image_dir, coco_annotation_file)
coco_data = coco_dataset.load_data()

nocaps_annotation_file = os.path.join(nocaps_dir, "annotations.json")
nocaps_dataset = NoCapsDataset(nocaps_annotation_file)
nocaps_data = nocaps_dataset.load_data()

# Split the data into train/validation/test sets
def split_data(data, train_ratio, val_ratio, test_ratio):
    random.shuffle(data)

    train_size = int(train_ratio * len(data))
    val_size = int(val_ratio * len(data))
    test_size = int(test_ratio * len(data))

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:train_size + val_size + test_size]

    return train_data, val_data, test_data

train_data, val_data, test_data = split_data(coco_data + nocaps_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

# Save the processed data
def save_data(data, file_path):
    serialized_data = []
    for image, caption in data:
        serialized_data.append({"image": image, "caption": caption})

    with open(file_path, "w") as file:
        json.dump(serialized_data, file)

save_data(train_data, os.path.join(data_dir, "train_data.json"))
save_data(val_data, os.path.join(data_dir, "val_data.json"))
save_data(test_data, os.path.join(data_dir, "test_data.json"))
