import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BlipTokenizer, BlipForConditionalGeneration, BlipConfig

# Set up paths
data_dir = "data"
preprocessed_data_path = os.path.join(data_dir, "preprocessed_data.json")
output_dir = "output"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
num_epochs = 10
learning_rate = 1e-4

# Load preprocessed data
with open(preprocessed_data_path, "r") as file:
    preprocessed_data = json.load(file)

# Define dataset and dataloader
class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        image_path = item["image_path"]
        caption = item["caption"]

        image = Image.open(image_path).convert("RGB")
        image = transforms.ToTensor()(image)

        return image, caption

dataset = ImageCaptionDataset(preprocessed_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load tokenizer and model configuration
tokenizer = BlipTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")
config = BlipConfig.from_pretrained("Salesforce/blip-image-captioning-base")
config.vocab_size = tokenizer.vocab_size

# Define the model
model = BlipForConditionalGeneration(config)
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0

    for images, captions in dataloader:
        images = images.to(device)
        captions = tokenizer(captions, padding=True, truncation=True, return_tensors="pt").input_ids
        captions = captions.to(device)

        outputs = model(input_ids=images, decoder_input_ids=captions[:, :-1])
        logits = outputs.logits

        loss = criterion(logits.view(-1, logits.size(-1)), captions[:, 1:].contiguous().view(-1))
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {average_loss}")

# Save the trained model
output_path = os.path.join(output_dir, "image_captioning_model.pt")
torch.save(model.state_dict(), output_path)
