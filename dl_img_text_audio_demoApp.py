import streamlit as st
from transformers import pipeline
from PIL import Image
import tempfile
from gtts import gTTS

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

story = None

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

             # Expand into a story
            text_generator = pipeline("text-generation")
            output = text_generator(imgDsp[0]['generated_text'], max_length=100, num_return_sequences=1)
            
            # Format
            story = (
                f"Once upon a time, {output[0]['generated_text'].lower()}, "
                "and what happened next transformed the world around it..."
            )

        st.subheader("📖 Story")
        st.write(story)

if st.button("🔊 Convert Story to Audio"):
    try:
        tts = gTTS(text=story, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            st.audio(tmp.name, format="audio/mp3")
    except Exception as e:
        st.error(f"Audio generation failed: {e}")

