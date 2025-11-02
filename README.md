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

# 3. Setup and Execution

To run this project on your local machine or a cloud instance, follow these steps.

---

### A. Data Acquisition Strategy (Choose ONE Method)

This project requires **PASCAL VOC 2012** (Segmentation) and **Flickr8k** (Captioning).

#### Option 1: Automated Download (Recommended for Training)

We use a single Python script (`data_download.ipynb`) to manage both datasets via the Kaggle API.

1.  **Kaggle API Key Setup (MANDATORY):**
    * Go to your **Kaggle profile** -> **Account** tab.
    * Download **`kaggle.json`** and place it in your local **`~/.kaggle/`** directory.

2.  **Execute Download Script:**
    * Run the file **`notebooks/data_download.ipynb`**. This script will:
        * Download and unzip **PASCAL VOC 2012** (Images & Masks).
        * Download and unzip **Flickr8k** (Images & Captions).
        * Place all files into the required **`data/`** directory structure.

#### Option 2: Shared Drive Access (Fastest for Inference/Demo)

If you have access to a shared Google Drive folder containing the pre-processed data and models (like the one you mentioned), follow these steps:

1.  **Add to My Drive:** Access the shared project folder and use the **"Add to My Drive"** or **"Add shortcut to Drive"** option. **CRITICALLY**, ensure the shortcut is placed in the **root** of your main Google Drive.
2.  **Mount Drive:** In your Colab/Jupyter notebook, run the code to mount Google Drive.
3.  **Update Paths:** Change the paths in **Cell 1** to point to the Drive location (e.g., `/content/drive/MyDrive/Shared_Project_Folder/...`).

Access Link : https://drive.google.com/drive/folders/1KQ7_V7fhGaBu--4dT0MGIPPTe8cCq4eI?usp=drive_link
---

### B. Pre-Trained Artifacts (Inference Ready)

For immediate inference, the following processed files must be present:

1.  Create a **`models/`** directory.
2.  Place the following files (generated after training) inside:
    * **`best_unet_model.pth`**
    * **`best_transformer_model.keras`**
    * **`tokenizer.pkl`** (Created from Flickr8k captions)
    * **`features.pkl`** (Xception features of Flickr8k images)

---
... (rest of the README)

### B. Pre-Trained Artifacts

The **Processed Data** and **Trained Models** are required for fast inference:
... (rest of the content remains the same)

### C. Local Installation & Run

```bash
# Clone the repository
git clone gh repo clone vp763/Image-Captioning-and-Segmentation

# Install all Python dependencies (e.g., from requirements.txt)
pip install -r requirements.txt

# Run the inference service (Assuming you use Flask/Streamlit/inference_service.py)
python inference_service.py
