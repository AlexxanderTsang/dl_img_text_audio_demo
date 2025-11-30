# --- Import ---
import requests
from PIL import Image
from google.colab import files
# Use a pipeline as a high-level helper
from transformers import pipeline
from IPython.display import Audio

# --- Step 1: Upload image ---
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Display image to verify
img = Image.open(image_path)

img_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
imgDsp = img_pipe(img)

# Create a text generation pipeline object
text_generator = pipeline("text-generation")
output = text_generator(imgDsp[0]['generated_text'], max_length=100, num_return_sequences=1)

pipe = pipeline("text-to-speech", model="facebook/mms-tts-eng")
output_speech = pipe(output[0]['generated_text'])

Audio(output_speech['audio'],rate=output_speech['sampling_rate'])
