from flask import Flask, jsonify, request
from flask_cors import CORS
import io
import base64
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For headless environments

# Define Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# -------------------------------
# Define your U-Net model here
# -------------------------------
class UNet(torch.nn.Module):
    # same UNet definition as before
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        def conv_block(in_c, out_c):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_c),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_c),
                torch.nn.ReLU(inplace=True)
            )
        self.encoder1 = conv_block(in_channels, 64)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.encoder2 = conv_block(64, 128)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.encoder3 = conv_block(128, 256)
        self.pool3 = torch.nn.MaxPool2d(2)
        self.encoder4 = conv_block(256, 512)
        self.pool4 = torch.nn.MaxPool2d(2)
        self.bottleneck = conv_block(512, 1024)
        self.upconv4 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        self.conv_last = torch.nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv_last(dec1))


# -------------------------------
# Inference Function
# -------------------------------
def unet_inference(image_bytes):
    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dim

    # Load model
    model = UNet()
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = output.squeeze().cpu().numpy() > 0.5  # Threshold to binary

    # Convert prediction to base64 PNG
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(pred_mask, cmap="gray")
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    base64_image = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return base64_image

# -------------------------------
# API Endpoint
# -------------------------------
@app.route('/api/inference', methods=['POST'])
def get_result():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    image_bytes = file.read()
    result_img = unet_inference(image_bytes)

    return jsonify({"mask": result_img})


# -------------------------------
# Run the app
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
