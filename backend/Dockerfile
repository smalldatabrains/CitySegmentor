# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a directory for the model
RUN mkdir -p /app/models && chmod 777 /app/models

# Copy only requirements first to leverage Docker cache
COPY requirements.txt ./

# Install dependencies with caching
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the application port (Flask default: 5000)
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
