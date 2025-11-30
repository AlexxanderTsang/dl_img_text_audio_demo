import streamlit as st
from transformers import pipeline
from PIL import Image

# -----------------------------
# Load transformer model locally
# -----------------------------

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
            img_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
            imgDsp = img_pipe(image)

            # Base caption from BLIP
            text_generator = pipeline("text-generation")
            output = text_generator(imgDsp[0]['generated_text'], max_length=100, num_return_sequences=1)

            # Expand into a story
            story = (
                f"Once upon a time, {output.lower()}, "
                "and what happened next transformed the world around it..."
            )

        st.subheader("📖 Story")
        st.write(story)
