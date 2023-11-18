import requests  # For making HTTP requests
from PIL import Image  # For image processing
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize the BlipProcessor for image captioning
link = "Salesforce/blip-image-captioning-base"
proc = BlipProcessor.from_pretrained(link)

# Initialize the BlipForConditionalGeneration model for image captioning
m = BlipForConditionalGeneration.from_pretrained(link)

# Specify the URL of the image to be captioned
img_url = 'https://i.im.ge/2023/05/03/UTf6DL.IMG-5355.jpg'

# Open and convert the image to RGB format
r_im = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Preprocess the image for caption generation
inputs = proc(r_im, return_tensors="pt")

# Generate captions for the image
out = m.generate(**inputs)

# Decode the generated captions and skip special tokens
caption = proc.decode(out[0], skip_special_tokens=True)

# Print the caption
print(caption)


# Convert the caption to speech
def cap_2_speech():
    tts = gTTS(text=caption_text, lang='en')
    tts.save('caption.mp3')

    # Initialize and load the audio file
    pygame.mixer.init()
    pygame.mixer.music.load('caption.mp3')

    # Play the audio
    pygame.mixer.music.play()

    # Wait for the audio playback to finish
    while pygame.mixer.music.get_busy():
        continue
