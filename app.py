"""
Streamlit app for Chest X-ray Pneumonia Detection (Binary: NORMAL vs PNEUMONIA)
- Expects a Keras model at models/pneumonia_model.h5 (change MODEL_PATH if needed)
- Preprocessing: resize to 150x150, rescale 1./255 (same as training)
- Shows probability, label, small bar chart, and optional Grad-CAM visualization
"""

from pathlib import Path
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import io

# ----------------- CONFIG -----------------
MODEL_PATH = Path("models") / "pneumonia_model.h5"
IMG_SIZE = (150, 150)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # 0 -> NORMAL, 1 -> PNEUMONIA
DEFAULT_THRESHOLD = 0.55
# ------------------------------------------

st.set_page_config(page_title="Chest X-ray Pneumonia Detector", layout="centered")

st.title("🩺 Chest X-ray Pneumonia Detector")
st.write(
    "Upload a chest X-ray image (JPEG/PNG) and the model will predict whether it is NORMAL or PNEUMONIA."
)

# --------- model loader with caching ---------
@st.cache_resource(show_spinner=False)
def load_trained_model(path: str):
    # Ensure we can load the model in the correct context
    tf.keras.backend.clear_session()
    if not Path(path).exists():
        return None, f"Model not found at: {path}"
    try:
        # Use compile=False for older TF versions to avoid optimizer import issues
        model = load_model(path, compile=False)
        # Warm up the model (prevents cold-start delay)
        dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        _ = model.predict(dummy)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {e}"

model, model_error = load_trained_model(str(MODEL_PATH))

if model_error:
    st.warning(model_error)
    st.info("Train the model (train_local.py) or place the trained model at models/pneumonia_model.h5.")
    st.stop()

# -------- utilities --------
def preprocess_pil(img: Image.Image, target_size=IMG_SIZE) -> np.ndarray:
    """Resize, convert to RGB and rescale to [0,1]. Returns batched array."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(model, pil_img: Image.Image):
    x = preprocess_pil(pil_img)
    # Set verbose=0 to suppress console output from predict in the Streamlit terminal
    pred = model.predict(x, verbose=0)[0][0]  # sigmoid output
    return float(pred)

def format_confidence(p):
    return f"{p*100:.2f}%"

# ---------- UI: sidebar ---------- 
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, float(DEFAULT_THRESHOLD), 0.01)
show_gradcam = st.sidebar.checkbox("Show Grad-CAM (slow)", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("Model path:")
st.sidebar.code(str(MODEL_PATH))

# ---------- UI: image input ----------
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload chest X-ray image", type=["jpg", "jpeg", "png"])

with col2:
    st.write("Model info")
    st.write(f"TF version: `{tf.__version__}`")
    gpus = tf.config.list_physical_devices("GPU")
    st.write("GPU available:" , bool(gpus))
    st.write("Threshold:", threshold)

# ---------- Load image to variable `pil_image` ----------
pil_image = None
if uploaded:
    try:
        pil_image = Image.open(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.error("Cannot open image. Make sure the file is a valid image.")
        st.stop()

if pil_image is None:
    st.info("Upload an image to see predictions.")
    st.stop()

st.image(pil_image, caption="Input Image", use_column_width=True)

# ---------- Prediction ----------
with st.spinner("Running prediction..."):
    try:
        pred_score = predict_image(model, pil_image)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

label_idx = int(pred_score > threshold)
label = CLASS_NAMES[label_idx]
confidence = pred_score if label_idx == 1 else (1.0 - pred_score)

# Show main result
colA, colB = st.columns([2, 1])
with colA:
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: **{format_confidence(confidence)}** (threshold = {threshold:.2f})")
    st.write("Raw model output (sigmoid):", f"{pred_score:.4f}")

with colB:
    # bar visualization
    prob_pos = pred_score
    prob_neg = 1.0 - prob_pos
    st.bar_chart({"NORMAL": [prob_neg], "PNEUMONIA": [prob_pos]})

# ---------- Grad-CAM (optional) ----------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """
    Produces a heatmap for binary classification using the final conv layer.
    Returns a heatmap resized to input size (values 0-1).
    """
    # Convert the input NumPy array back to a TensorFlow Tensor
    # with gradient tracking explicitly enabled for Grad-CAM to work.
    x = tf.convert_to_tensor(img_array)
    
    try:
        # find last conv layer if not provided
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer_name = layer.name
                    break
        if last_conv_layer_name is None:
            return None, "No Conv2D layer found in model."

        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            # Watch the input tensor for gradient calculation
            tape.watch(x) 
            conv_outputs, preds = grad_model(x)
            # Loss is defined as the score of the predicted class (index 0 is NORMAL, index 1 is PNEUMONIA)
            # Since this is a binary classifier, we use index 0 (Pneumonia score)
            loss = preds[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None, "Could not compute gradients."

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE)
        heatmap = tf.squeeze(heatmap).numpy()
        return heatmap, None
    except Exception as e:
        return None, str(e)

if show_gradcam:
    with st.spinner("Computing Grad-CAM... (may be slow)"):
        try:
            x = preprocess_pil(pil_image)
            heatmap, err = make_gradcam_heatmap(x, model)
            if heatmap is None:
                st.warning("Grad-CAM not available: " + (err or "unknown reason"))
            else:
                # overlay heatmap on image
                import matplotlib.pyplot as plt
                import matplotlib.cm as cm
                # Matplotlib import fix for older versions
                fig, ax = plt.subplots(figsize=(6,6))
                ax.imshow(pil_image.resize(IMG_SIZE))
                cmap = cm.get_cmap("jet")
                ax.imshow(heatmap, cmap=cmap, alpha=0.4, extent=(0, IMG_SIZE[0], IMG_SIZE[1], 0))
                ax.axis("off")
                st.pyplot(fig)
        except Exception as e:
            st.warning("Grad-CAM failed: " + str(e))

# ---------- Footer / tips ----------
st.markdown("---")
st.markdown("**Tips**:")
st.markdown(
    "- Best results when uploading frontal chest X-rays (JPEG/PNG). "
    "- If the model gives unexpected results, try multiple images or adjust the decision threshold. "
)
st.markdown("**Note:** This demo uses a model you trained locally. For production, consider server-side model hosting, input validation, and clinical validation.")