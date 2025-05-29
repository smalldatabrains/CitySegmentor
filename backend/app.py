from flask import Flask, jsonify, request
from flask_cors import CORS
import io
import base64
from PIL import Image
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For headless servers
import logging
import os

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins":"*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "expose_headers": ["Content-Range", "X-Content-Range"],
        "max_age": 1728000,
        "supports_credentials": False
    }
})

# Define model paths
MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

logger.info("Starting model loading...")
# Download and save model locally if not already present
if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    logger.info(f"Downloading model {MODEL_ID} to {MODEL_DIR}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    processor.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    logger.info("Model downloaded and saved successfully!")
else:
    logger.info("Loading model from local directory...")
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR)

model.eval()
logger.info("Model loaded successfully!")

# Cityscapes labels (19 classes)
CITYSCAPES_LABELS = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle"
]

# Use a consistent color palette (e.g., tab20 from matplotlib)
import matplotlib.cm as cm
def get_color_palette(num_classes=19):
    cmap = cm.get_cmap("tab20", num_classes)
    return [tuple((np.array(cmap(i)[:3]) * 255).astype(int)) for i in range(num_classes)]

PALETTE = get_color_palette(len(CITYSCAPES_LABELS))

def segment_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )

    pred = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    # Create RGB mask
    rgb_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(PALETTE):
        rgb_mask[pred == idx] = color

    mask_image = Image.fromarray(rgb_mask)

    # Convert to base64 PNG
    buf = io.BytesIO()
    mask_image.save(buf, format="PNG")
    buf.seek(0)
    base64_mask = base64.b64encode(buf.read()).decode("utf-8")

    # Prepare the legend
    legend = [
        {"label": label, "color": f"rgb({r},{g},{b})"}
        for label, (r, g, b) in zip(CITYSCAPES_LABELS, PALETTE)
    ]

    return base64_mask, legend

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/api/inference', methods=['POST', 'OPTIONS'])
def inference():
    if request.method == 'OPTIONS':
        return '', 204
    
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400
    
    file = request.files['image']
    image_bytes = file.read()

    try:
        mask_base64, legend = segment_image(image_bytes)
        return jsonify({
            "segmentation_mask": mask_base64,
            "legend": legend
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
