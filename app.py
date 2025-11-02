# ===========================================
# APP.PY — Google Drive + Colab + pyngrok + UI (FINAL)
# ===========================================

import os
import pickle
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
import segmentation_models_pytorch as smp
import tensorflow as tf
from keras.applications.xception import Xception, preprocess_input
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pyngrok import ngrok

# --- 🔥 Google Drive Model Paths ---
UNET_CKPT_PATH = r"/content/drive/MyDrive/ZidioProject/best_unet_model.pth"
TRANSFORMER_MODEL_PATH = r"/content/drive/My Drive/Image_Captioning/best_transformer_model.keras"
TOKENIZER_PATH = r"/content/drive/My Drive/Image_Captioning/tokenizer.pkl"

MAX_LENGTH = 34
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Flask Setup ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Global Model Variables ---
model_seg = None
caption_model = None
xception_model = None
caption_tokenizer = None

# --- Load Models ---
def load_all_models():
    global model_seg, caption_model, xception_model, caption_tokenizer
    try:
        print("🔄 Loading Segmentation Model...")
        model_seg = smp.Unet(
            encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1
        ).to(DEVICE)
        checkpoint = torch.load(UNET_CKPT_PATH, map_location=DEVICE)
        model_seg.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model_seg.eval()
        print("✅ Segmentation Model Loaded.")

        print("🔄 Loading Captioning Model & Tokenizer...")
        caption_model = load_model(TRANSFORMER_MODEL_PATH, compile=False)
        xception_model = Xception(include_top=False, pooling="avg")
        with open(TOKENIZER_PATH, "rb") as f:
            caption_tokenizer = pickle.load(f)
        print("✅ Captioning Model Loaded.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load models — {e}")

load_all_models()

# --- Helper Functions ---
def preprocess_image_seg(image_path, img_size=(256, 256)):
    image = Image.open(image_path).convert("RGB").resize(img_size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def generate_mask(model, image_path, output_filename):
    input_tensor_seg = preprocess_image_seg(image_path)
    with torch.no_grad():
        seg_output = model(input_tensor_seg.to(DEVICE))
        seg_output = torch.sigmoid(seg_output).cpu().numpy()[0, 0]
    mask_display = (cv2.resize(seg_output, (256, 256)) * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_display, 'L')
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    mask_img.save(mask_path)
    return mask_path

def idx_to_word(integer, tokenizer):
    return tokenizer.index_word.get(integer)

def extract_caption_features(filename, xception_model, img_size=(299, 299)):
    image = load_img(filename, target_size=img_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = xception_model.predict(image, verbose=0)
    return feature.reshape(1, -1)

def generate_caption(model, tokenizer, photo_features, max_length):
    in_text = "startseq"
    photo_features_reshaped = photo_features.reshape(1, -1)
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([tf.constant(photo_features_reshaped, dtype=tf.float32),
                              tf.constant(sequence, dtype=tf.int32)], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None or word == "endseq":
            break
        in_text += " " + word
    return in_text.replace("startseq", "").strip()

# --- HTML Template ---
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>AI Caption + Segmentation</title>
  <style>
    body { font-family: Arial; text-align: center; background: #f7f7f7; }
    .container { margin-top: 50px; }
    img { max-width: 45%; margin: 10px; border-radius: 10px; box-shadow: 0 0 10px #aaa; }
    h2 { color: #333; }
  </style>
</head>
<body>
  <div class="container">
    <h1>🧠 Image Captioning + Segmentation</h1>
    <form method="POST" enctype="multipart/form-data" action="/analyze">
      <input type="file" name="file" accept="image/*" required>
      <button type="submit">Analyze</button>
    </form>
    {% if filename %}
      <h2>Caption:</h2>
      <p><b>{{ caption }}</b></p>
      <img src="/uploads/{{ filename }}" alt="Original">
      <img src="/uploads/{{ mask_filename }}" alt="Mask">
    {% endif %}
  </div>
</body>
</html>
"""

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_PAGE)

from flask import send_from_directory

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "No file uploaded!"

    file = request.files['file']
    if file.filename == '':
        return "No file selected!"

    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return "Invalid file type!"

    original_filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    file.save(filepath)

    mask_filename = "mask_" + original_filename
    caption = "Error: Model not loaded."

    if model_seg:
        generate_mask(model_seg, filepath, mask_filename)

    if caption_model and xception_model and caption_tokenizer:
        photo_features = extract_caption_features(filepath, xception_model)
        caption = generate_caption(caption_model, caption_tokenizer, photo_features, MAX_LENGTH)

    return render_template_string(HTML_PAGE, filename=original_filename, mask_filename=mask_filename, caption=caption)

# --- Run Flask + ngrok ---
if __name__ == '__main__':
    print("🚀 Starting Flask with ngrok tunnel...")
    public_url = ngrok.connect(5000).public_url
    print(f"🌍 Public URL: {public_url}")
    print("✅ Open this link in your browser to access the app.")
    app.run(port=5000)
