# 🧠 Image Segmentation and Focused Captioning (Transformer-U-Net Fusion)

**Project Goal:** To create an end-to-end Deep Learning pipeline that first **segments a specific object** (like a dog or a car) from an image and then generates a **contextually focused caption** *only* for the segmented area. This project successfully fuses two complex models: **PyTorch U-Net** and a **TensorFlow Transformer** architecture.

---

## 🌟 1. Key Features & Technologies

* **Deep Fusion Architecture:** Combines a **PyTorch U-Net** (for precise pixel-level masking) with a **Keras Transformer** (for sequence generation).  

[Image of Transformer Architecture]

* **Focused Captioning:** Caption generation is biased towards the segmented object, enhancing relevance and description accuracy.
* **Best-in-Class Segmentation:** Achieved a **Dice Score of [PLACEHOLDER: 0.85+]** on the PASCAL VOC 2012 test set using a **ResNet-34 pre-trained encoder**. (Segmentation Accuracy: [PLACEHOLDER: 90+]%).
* **Modern NLP:** Used a **Multi-Head Attention Transformer** (Self-Attention based) for the caption generation, outperforming traditional RNN/LSTM models.
* **Zero-API Local Setup:** Designed for quick deployment; requires no external API key for local execution.

| Technology | Role |
| :--- | :--- |
| **PyTorch** | U-Net Segmentation Model (ResNet34 Backbone) |
| **TensorFlow/Keras**| Transformer Model (Seq2Seq) and Xception (Image Encoder) |
| **PASCAL VOC 2012** | Segmentation Training Data |
| **Flickr8k** | Captioning Training Data |

---

## 🚀 2. Architecture and Pipeline (The Fusion)

Our pipeline works like a **"Photographer and a Poet"**:

1.  **Photographer (U-Net):** Takes the raw image, identifies the target object, and produces a **pixel mask** (Dice Score: 0.85+).
2.  **The Bridge (Python/Numpy):** The original image is multiplied by the generated mask, effectively **blacking out the background** and isolating the object. 
3.  **The Encoder (Xception):** Extracts features (2048-dim vector) from the **masked image**.
4.  **The Poet (Transformer):** The Transformer's **Decoder** uses the feature vector and generates the most relevant **sentence**.

### 🔍 Results Snapshot (Segmentation & Captioning)

| Metric | Validation Set Value |
| :--- | :--- |
| **Segmentation Dice Score** | **0.[PLACEHOLDER: 8553]** |
| **Captioning Val Loss (Final)**| [PLACEHOLDER: e.g., 2.89] |
| **BLEU-4 Score (Test Set)** | [PLACEHOLDER: e.g., 0.25 (Optional)] |

---

## 💡 3. Setup and Execution

To run this project on your local machine or a cloud instance, follow these steps. **(No Kaggle/Drive setup needed post-training!)**

### A. Data Preparation (Initial Setup Only)

1.  Create a **`data/`** directory in the project root.
2.  Inside `data/`, place the **Flickr8k Dataset** in the structure expected by the code:
    ```
    /data/
    ├── Flickr8k_Dataset/  (All .jpg files)
    └── Flickr8k_text/
        └── Flickr8k.token.txt
    ```

### B. Pre-Trained Artifacts

The **Processed Data** and **Trained Models** are required for inference:

1.  Create a **`models/`** directory.
2.  Place the following files (generated during the training phase) inside:
    * **`best_unet_model.pth`**
    * **`best_transformer_model.keras`**
    * **`tokenizer.pkl`**
    * **`features.pkl`** (Processed image features - **CRITICAL** for fast loading)

### C. Local Installation & Run

```bash
# Clone the repository
git clone [PLACEHOLDER: YOUR_REPO_LINK]
cd [PLACEHOLDER: YOUR_REPO_NAME]

# Install all Python dependencies (e.g., from requirements.txt)
pip install -r requirements.txt

# Run the inference service (Assuming you use Flask/Streamlit/inference_service.py)
python inference_service.py
