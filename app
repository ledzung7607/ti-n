import streamlit as st
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model

# ==========================
# CẤU HÌNH TRANG
# ==========================
st.set_page_config(
    page_title="Vietnamese Banknote Recognition",
    page_icon="💵",
    layout="wide"
)

# ==========================
# CSS
# ==========================
st.markdown("""
<style>

.main {
    background-color: #f5fff5;
}

.title {
    text-align:center;
    color:#0b8f3c;
    font-size:40px;
    font-weight:bold;
}

.subtitle {
    text-align:center;
    color:gray;
    font-size:18px;
}

.result-box{
    padding:20px;
    border-radius:15px;
    background:#e8fff0;
    border:2px solid #0b8f3c;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
model = load_model("money.keras")

with open("class_indices.json","r") as f:
    class_indices = json.load(f)

classes = {v:k for k,v in class_indices.items()}

# ==========================
# HEADER
# ==========================
st.markdown(
    '<div class="title">💵 Vietnamese Banknote Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">CNN Deep Learning Project</div>',
    unsafe_allow_html=True
)

st.write("")
st.write("")

uploaded_file = st.file_uploader(
    "📤 Upload Banknote Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    col1, col2 = st.columns([1,1])

    with col1:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        img = image.resize((200,200))

        img = np.array(img)

        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)

        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)

        class_id = np.argmax(prediction)

        confidence = float(np.max(prediction)*100)

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.success(
            f"Predicted Banknote: {classes[class_id]} VND"
        )

        st.progress(int(confidence))

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

st.markdown("---")

st.markdown("""
### 📖 About

This application uses a Convolutional Neural Network (CNN)
to recognize Vietnamese banknotes from uploaded images.

Supported banknotes depend on the classes used during training.
""")
