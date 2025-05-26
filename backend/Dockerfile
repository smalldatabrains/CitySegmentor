FROM python:3.9.18-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model during build
RUN python -c "import torch; from transformers import AutoImageProcessor, SegformerForSemanticSegmentation; \
    torch.device('cpu'); \
    processor = AutoImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-cityscapes-1024-1024'); \
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-cityscapes-1024-1024')"

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
