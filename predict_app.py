import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from classifier import CNNClassifier
import os
from tkinterdnd2 import DND_FILES, TkinterDnD

class PredictApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rust Detection Classifier")
        self.root.geometry("600x400")
        
        # Load the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNNClassifier().to(self.device)
        
        # Load the trained weights
        checkpoint = torch.load('best_model.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Drop zone
        self.drop_label = ttk.Label(
            self.main_frame,
            text="Drag and drop an image here",
            padding="50",
            relief="solid"
        )
        self.drop_label.grid(row=0, column=0, pady=20, sticky=(tk.W, tk.E))
        
        # Enable drag and drop
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.handle_drop)
        
        # Image preview
        self.preview_label = ttk.Label(self.main_frame)
        self.preview_label.grid(row=1, column=0, pady=10)
        
        # Result label
        self.result_label = ttk.Label(
            self.main_frame,
            text="Prediction will appear here",
            font=('Arial', 12, 'bold')
        )
        self.result_label.grid(row=2, column=0, pady=10)
        
    def handle_drop(self, event):
        file_path = event.data
        
        # Clean up the file path (remove curly braces if present)
        file_path = file_path.strip('{}')
        
        if os.path.isfile(file_path):
            # Load and display the image
            image = Image.open(file_path)
            
            # Create preview
            preview = image.copy()
            preview.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(preview)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
            
            # Prepare image for model
            img_tensor = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(img_tensor)
                _, predicted = torch.max(outputs.data, 1)
                
                # Get probability
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence = probabilities[0][predicted.item()].item() * 100
                
                # Update result label
                result_text = f"Prediction: {'Rust' if predicted.item() == 1 else 'No Rust'}\nConfidence: {confidence:.2f}%"
                self.result_label.configure(text=result_text)

def main():
    root = TkinterDnD.Tk()
    app = PredictApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
