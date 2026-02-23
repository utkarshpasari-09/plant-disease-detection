import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="🌿",
    layout="wide"
)

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(120deg,#f6fff9,#e9f7ef);
}

/* Top Navigation */
.navbar {
    background: white;
    padding: 18px 40px;
    border-radius: 12px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Hero Section */
.hero {
    padding: 30px;
    background: white;
    border-radius: 18px;
    box-shadow: 0px 6px 30px rgba(0,0,0,0.06);
}

/* Prediction Card */
.result-card {
    padding: 25px;
    background: white;
    border-radius: 18px;
    box-shadow: 0px 6px 30px rgba(0,0,0,0.08);
}

/* Buttons */
.stButton>button {
    background-color:#2d6a4f;
    color:white;
    font-size:18px;
    border-radius:10px;
    padding:10px 26px;
    border:none;
}
.stButton>button:hover {
    background:#1b4332;
}

/* Upload */
.upload-box {
    padding:20px;
    border-radius:14px;
    border:2px dashed #95d5b2;
    background:#ffffff;
}

footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model.keras")

model = load_model()

# ---------------- CLASS LABELS ----------------
class_names = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust',
'Apple___healthy','Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew',
'Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
'Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight',
'Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight',
'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus',
'Tomato___healthy']

# ---------------- PREDICTION FUNCTION ----------------
def predict(image):
    img = image.resize((128,128))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)
    index = np.argmax(pred)
    confidence = float(np.max(pred))

    return class_names[index], confidence

# ---------------- TOP NAVIGATION ----------------
st.markdown('<div class="navbar">', unsafe_allow_html=True)
page = st.radio(
    "Navigation",
    ["Home", "About", "Detect Disease"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- HOME ----------------
if page == "Home":
    st.markdown('<div class="hero">', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2,1])

    with col1:
        st.title("🌿 AI-Based Plant Disease Detection")
        st.write("""
This system uses **Deep Learning (CNN)** to identify plant diseases from leaf images.

✔ Upload a leaf image  
✔ AI analyzes symptoms  
✔ Get instant diagnosis  
✔ Helps farmers act early  

Designed to improve crop health and reduce yield loss.
        """)

    with col2:
        st.image("home_page.jpeg", width="stretch")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ABOUT ----------------
elif page == "About":
    st.markdown('<div class="hero">', unsafe_allow_html=True)

    st.header("📊 About the Project")

    st.write("""
This project uses a **Convolutional Neural Network trained on 87,000+ images**
across **38 plant disease classes**.

### Dataset Includes:
- 70,295 Training Images  
- 17,572 Validation Images  
- Multiple crop species and disease types  

### Goal:
To provide an accessible AI tool for **early disease detection** in agriculture.
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- DETECTION ----------------
elif page == "Detect Disease":

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, width="stretch")

            if st.button("Analyze"):
                with st.spinner("Running Model..."):
                    label, conf = predict(img)

                if "healthy" in label.lower():
                    st.success(f"Healthy Plant ✅")
                else:
                    st.error(f"Disease Detected: {label}")

                st.progress(conf)
                st.write(f"Confidence: **{conf*100:.2f}%**")

        else:
            st.info("Upload an image to start analysis.")

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("AI Agriculture Assistant • Built with TensorFlow + Streamlit")