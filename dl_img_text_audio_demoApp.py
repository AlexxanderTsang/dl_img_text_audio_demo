import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# -----------------------------
# Load transformer model locally
# -----------------------------
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

processor, model = load_model()

st.title("📘 Image → Story Generator (Local Transformer Model)")

# -----------------------------
# UI: click button to choose file
# -----------------------------
uploaded_file = None

if st.button("📁 Select an Image"):
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# If user uploads normally without button (allowed)
if not uploaded_file:
    uploaded_file = st.file_uploader("Or upload directly:", type=["jpg", "jpeg", "png"])

# -----------------------------
# Once file is selected → display it
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Selected Image", use_column_width=True)

    # -----------------------------
    # Generate story
    # -----------------------------
    if st.button("Generate Story"):
        with st.spinner("Creating story..."):
            inputs = processor(image, return_tensors="pt")

            # Base caption from BLIP
            caption_ids = model.generate(**inputs, max_length=50)
            caption = processor.decode(caption_ids[0], skip_special_tokens=True)

            # Expand into a story
            story = (
                f"Once upon a time, {caption.lower()}, "
                "and what happened next transformed the world around it..."
            )

        st.subheader("📖 Story")
        st.write(story)
