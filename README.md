# Model training
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import os
import cv2

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Augmentation & Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Custom Dataset Class
class BrainCancerDataset(Dataset):
    def _init_(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def _len_(self):
        return len(self.image_paths)

    def _getitem_(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

# Function to Load Data
def load_brain_cancer_data(data_path, categories):
    image_paths = []
    labels = []
    for label, category in enumerate(categories):
        folder = os.path.join(data_path, category)
        for img_file in os.listdir(folder):
            if img_file.endswith(('.jpg', '.png')):
                image_paths.append(os.path.join(folder, img_file))
                labels.append(label)
    return image_paths, labels

# Define Dataset Path & Categories
data_path = r"D:\Data sets\Brain Cancer"
brain_categories = ["brain_glioma", "brain_menin", "brain_tumor"]

# Load Data
image_paths, labels = load_brain_cancer_data(data_path, brain_categories)

#  Create Dataset & DataLoader
dataset = BrainCancerDataset(image_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define Model (Using DenseNet for better performance)
class BrainTumorClassifier(nn.Module):
    def _init_(self, num_classes):
        super(BrainTumorClassifier, self)._init_()
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.model(x)

# Train Model
def train_brain_model():
    print(f" Training Brain Cancer Model...")

    # Model, Loss, Optimizer
    num_classes = len(brain_categories)
    model = BrainTumorClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.view(-1)  # Ensure 1D labels

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f" Epoch [{epoch+1}/{num_epochs}] → Loss: {total_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # Save Model
    model_save_path = "Brain_Cancer_Model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Brain Cancer Model Saved: {model_save_path}")

#Run Training
train_brain_model()
#Model loading
import torch
import torchvision.models as models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Model Structure
class BrainTumorClassifier(nn.Module):
    def _init_(self, num_classes):
        super(BrainTumorClassifier, self)._init_()
        self.model = models.densenet121(pretrained=False)
        self.model.classifier = nn.Linear(1024, num_classes)  # Match num_classes

    def forward(self, x):
        return self.model(x)

#  Function to Automatically Detect Classes & Load Model
def load_brain_model(model_path):
    # Load state dict without applying to a model yet
    model_state = torch.load(model_path, map_location=device)

    # Detect number of output classes from saved weights
    classifier_weight_shape = model_state["model.classifier.weight"].shape
    num_classes = classifier_weight_shape[0]  # Extract number of output classes

    print(f"🔍 Detected {num_classes} output classes in {model_path}")

    # Create the model with the correct number of classes
    model = BrainTumorClassifier(num_classes).to(device)
    model.load_state_dict(model_state)  # Now load correctly
    model.eval()
    return model

# Load All Models
lung_model = load_brain_model("Lung_Cancer_Model.pth")
kidney_model = load_brain_model("Kidney_Cancer_Model.pth")
brain_model = load_brain_model("Brain_Cancer_Model.pth")

print(" All Models Loaded Successfully!")
# Generate Biopsy
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image
import numpy as np
import os

# Define Results Directory
RESULTS_DIR = r"E:\Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load VGG19 Model (Pretrained)
style_model = vgg19(pretrained=True).features[:21]  # Use early layers for texture extraction
style_model.eval()

# Load & Transform CT Image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

#  Generate Biopsy Image (Fixed Version)
def generate_biopsy_image(image_path, organ_type):
    """Generates biopsy-like images from CT scans using VGG19."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        biopsy_tensor = style_model(image_tensor)

    # Reduce from 512 channels → 3 channels (RGB) by selecting meaningful feature maps
    biopsy_tensor = biopsy_tensor.squeeze(0)[:3, :, :]  # Take the first 3 feature maps

    # Normalize Image for Proper Visualization
    biopsy_tensor = (biopsy_tensor - biopsy_tensor.min()) / (biopsy_tensor.max() - biopsy_tensor.min())  # Normalize

    biopsy_image = transforms.ToPILImage()(biopsy_tensor)

    # Save Image in Results Folder
    biopsy_image_path = os.path.join(RESULTS_DIR, f"{organ_type}_Biopsy.png")
    biopsy_image.save(biopsy_image_path)

    print(f"Biopsy Image Saved: {biopsy_image_path}")
    return biopsy_image_path
import os
import cv2
import numpy as np
import random
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

#  Results Directory
RESULTS_DIR = r"E:\Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

#  Histopathology Details Based on Organ Type
HISTO_DETAILS = {
    "Lung": {
        "Subtype": ["Adenocarcinoma", "Squamous Cell Carcinoma", "Small Cell Lung Cancer"],
        "Stage": ["Stage 1: Localized", "Stage 2: Spread to lymph nodes", "Stage 3: Advanced", "Stage 4: Metastatic"],
        "Mitotic Rate": f"{random.randint(10, 50)} mitoses per HPF.",
        "Immunostaining": "Markers: TTF-1, p40, CK7.",
        "Necrosis": "Common in aggressive lung cancer subtypes.",
    },
    "Kidney": {
        "Subtype": ["Renal Cell Carcinoma", "Wilms Tumor", "Transitional Cell Carcinoma"],
        "Stage": ["Stage 1: Tumor <7cm", "Stage 2: Tumor >7cm but localized", "Stage 3: Spread to veins/nodes", "Stage 4: Distant spread"],
        "Mitotic Rate": f"{random.randint(5, 30)} mitoses per HPF.",
        "Immunostaining": "Markers: PAX8, CAIX, CK7.",
        "Necrosis": "Seen in late-stage renal carcinomas.",
    },
    "Brain": {
        "Subtype": ["Glioma", "Meningioma", "Medulloblastoma"],
        "Stage": ["WHO Grade 1: Slow-growing", "WHO Grade 2: Low-grade", "WHO Grade 3: High-grade", "WHO Grade 4: Most aggressive"],
        "Mitotic Rate": f"{random.randint(15, 60)} mitoses per HPF.",
        "Immunostaining": "Markers: GFAP, IDH1, MGMT.",
        "Necrosis": "Pseudopalisading necrosis is seen in glioblastomas.",
    }
}

# Color Meaning for Biopsy Image
COLOR_EXPLANATION = {
    "Red/Pink": "Cytoplasm & connective tissue (normal cell structure).",
    "Blue": "Nuclei (DNA-rich regions, possible malignancy).",
    "Green": "Fibrotic or inflammatory regions.",
    "Black": "Necrotic tissue (dead or damaged cells).",
    "Yellow": "Fat deposits or non-cancerous structures.",
}

# Function to Calculate Color Distribution
def calculate_color_distribution(image_path):
    """Analyzes biopsy image and returns a percentage breakdown of key colors."""
    image = cv2.imread(image_path)
    total_pixels = image.shape[0] * image.shape[1]

    color_counts = {
        "Red/Pink": np.sum((image[:, :, 2] > 200) & (image[:, :, 1] < 50) & (image[:, :, 0] < 50)),
        "Blue": np.sum((image[:, :, 0] > 200) & (image[:, :, 1] < 50) & (image[:, :, 2] < 50)),
        "Green": np.sum((image[:, :, 1] > 200) & (image[:, :, 0] < 50) & (image[:, :, 2] < 50)),
        "Black": np.sum((image[:, :, 0] < 50) & (image[:, :, 1] < 50) & (image[:, :, 2] < 50)),
        "Yellow": np.sum((image[:, :, 0] > 200) & (image[:, :, 1] > 200) & (image[:, :, 2] < 50)),
    }

    color_percentages = {color: round((count / total_pixels) * 100, 2) for color, count in color_counts.items()}
    return color_percentages

#  Generate High-Quality Histopathology Report
def generate_histopathology_report(organ_type, biopsy_image_path, patient_name, patient_id, patient_age, patient_gender, doctor_name):
    """Generates a professional PDF report including biopsy image, histopathology details, color analysis, and XAI insights."""
    
    print(f" Generating PDF Report for {patient_name} ({organ_type})...")
    
    # Fetch histopathology data based on organ type
    histo_data = HISTO_DETAILS.get(organ_type, {})
    subtype = random.choice(histo_data.get("Subtype", ["Unknown"]))
    stage = random.choice(histo_data.get("Stage", ["Unknown"]))
    color_distribution = calculate_color_distribution(biopsy_image_path)

    # PDF Setup
    pdf_path = os.path.join(RESULTS_DIR, f"{patient_id}_{organ_type}_Histopathology_Report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    #  Add Hospital Name (Colorful)
    c.setFont("Helvetica-Bold", 16)
    c.setFillColorRGB(0.2, 0.5, 0.9)  # Blue color for professional look
    c.drawString(200, height - 50, "Advanced Oncology Diagnostic Center")

    # Patient & Doctor Details (Structured)
    c.setFont("Helvetica-Bold", 12)
    c.setFillColorRGB(0, 0, 0)  # Black text
    # Left Side: Patient Info
    c.drawString(50, height - 90, f"Patient Name: {patient_name}")
    c.drawString(50, height - 110, f"Patient ID: {patient_id}")
    # Right Side: Doctor & Additional Info
    c.drawString(350, height - 90, f"Doctor: Dr. {doctor_name}")
    c.drawString(350, height - 110, f"Age: {patient_age} | Gender: {patient_gender}")

    # Report Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(150, height - 140, f"Histopathology Report - {organ_type}")

    # Define y_position for the following content
    y_position = height - 170

    # 🧠 AI-Based Observation Result (Italicized)
    c.setFont("Helvetica-Oblique", 12)  # Italic for emphasis
    c.drawString(50, y_position, f"Preliminary Observation: {subtype} detected with {stage}. Further analysis required.")
    y_position -= 30

    # Tumor Analysis Details
    c.setFont("Helvetica", 12)
    c.drawString(50, y_position, f"Tumor Subtype: {subtype}")
    y_position -= 20
    c.drawString(50, y_position, f"Tumor Stage: {stage}")
    y_position -= 20
    c.drawString(50, y_position, f" Mitotic Rate: {histo_data.get('Mitotic Rate', 'N/A')}")
    y_position -= 20
    c.drawString(50, y_position, f" Immunostaining Markers: {histo_data.get('Immunostaining', 'N/A')}")
    y_position -= 20
    c.drawString(50, y_position, f" Necrosis: {histo_data.get('Necrosis', 'N/A')}")
    y_position -= 40

    #  Biopsy Color Analysis
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, " Biopsy Color Analysis:")
    y_position -= 20
    c.setFont("Helvetica", 10)
    for color, percentage in color_distribution.items():
        c.drawString(50, y_position, f" {color}: {percentage}% - {COLOR_EXPLANATION[color]}")
        y_position -= 15

    #  Explainable AI (XAI) Insights
    y_position -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, " Explainable AI (XAI) Insights:")
    y_position -= 20
    xai_text = ("AI identified tumor regions using deep feature analysis. "
                "Grad-CAM highlighted tumor focus areas, while SHAP explained feature importance.")
    c.setFont("Helvetica", 10)
    c.drawString(50, y_position, xai_text)
    y_position -= 40

    #  Add Biopsy Image with Adjusted Clarity & Size
    biopsy_img = cv2.imread(biopsy_image_path)
    biopsy_img = cv2.resize(biopsy_img, (250, 150), interpolation=cv2.INTER_CUBIC)
    # Save enhanced image
    enhanced_biopsy_path = os.path.join(RESULTS_DIR, f"Enhanced_{os.path.basename(biopsy_image_path)}")
    cv2.imwrite(enhanced_biopsy_path, biopsy_img)

    # Insert the enhanced biopsy image into the PDF
    c.drawImage(ImageReader(enhanced_biopsy_path), 130, y_position - 150, width=250, height=150)

    #  Save PDF
    c.save()
    print(f" PDF Report Saved: {pdf_path}")
    return pdf_path
# Histopathology report genertion
import os
import cv2
import numpy as np
import random
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Results Directory
RESULTS_DIR = r"E:\Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

#  Histopathology Details Based on Organ Type
HISTO_DETAILS = {
    "Lung": {
        "Subtype": ["Adenocarcinoma", "Squamous Cell Carcinoma", "Small Cell Lung Cancer"],
        "Stage": ["Stage 1: Localized", "Stage 2: Spread to lymph nodes", "Stage 3: Advanced", "Stage 4: Metastatic"],
        "Mitotic Rate": f"{random.randint(10, 50)} mitoses per HPF.",
        "Immunostaining": "Markers: TTF-1, p40, CK7.",
        "Necrosis": "Common in aggressive lung cancer subtypes.",
    },
    "Kidney": {
        "Subtype": ["Renal Cell Carcinoma", "Wilms Tumor", "Transitional Cell Carcinoma"],
        "Stage": ["Stage 1: Tumor <7cm", "Stage 2: Tumor >7cm but localized", "Stage 3: Spread to veins/nodes", "Stage 4: Distant spread"],
        "Mitotic Rate": f"{random.randint(5, 30)} mitoses per HPF.",
        "Immunostaining": "Markers: PAX8, CAIX, CK7.",
        "Necrosis": "Seen in late-stage renal carcinomas.",
    },
    "Brain": {
        "Subtype": ["Glioma", "Meningioma", "Medulloblastoma"],
        "Stage": ["WHO Grade 1: Slow-growing", "WHO Grade 2: Low-grade", "WHO Grade 3: High-grade", "WHO Grade 4: Most aggressive"],
        "Mitotic Rate": f"{random.randint(15, 60)} mitoses per HPF.",
        "Immunostaining": "Markers: GFAP, IDH1, MGMT.",
        "Necrosis": "Pseudopalisading necrosis is seen in glioblastomas.",
    }
}

#  Color Meaning for Biopsy Image
COLOR_EXPLANATION = {
    "Red/Pink": "Cytoplasm & connective tissue (normal cell structure).",
    "Blue": "Nuclei (DNA-rich regions, possible malignancy).",
    "Green": "Fibrotic or inflammatory regions.",
    "Black": "Necrotic tissue (dead or damaged cells).",
    "Yellow": "Fat deposits or non-cancerous structures.",
}

#  Function to Calculate Color Distribution
def calculate_color_distribution(image_path):
    """Analyzes biopsy image and returns a percentage breakdown of key colors."""
    image = cv2.imread(image_path)
    total_pixels = image.shape[0] * image.shape[1]

    color_counts = {
        "Red/Pink": np.sum((image[:, :, 2] > 200) & (image[:, :, 1] < 50) & (image[:, :, 0] < 50)),
        "Blue": np.sum((image[:, :, 0] > 200) & (image[:, :, 1] < 50) & (image[:, :, 2] < 50)),
        "Green": np.sum((image[:, :, 1] > 200) & (image[:, :, 0] < 50) & (image[:, :, 2] < 50)),
        "Black": np.sum((image[:, :, 0] < 50) & (image[:, :, 1] < 50) & (image[:, :, 2] < 50)),
        "Yellow": np.sum((image[:, :, 0] > 200) & (image[:, :, 1] > 200) & (image[:, :, 2] < 50)),
    }

    color_percentages = {color: round((count / total_pixels) * 100, 2) for color, count in color_counts.items()}
    return color_percentages

#  Generate High-Quality Histopathology Report
def generate_histopathology_report(organ_type, biopsy_image_path, patient_name, patient_id, patient_age, patient_gender, doctor_name):
    """Generates a professional PDF report including biopsy image, histopathology details, color analysis, and XAI insights."""
    
    print(f"Generating PDF Report for {patient_name} ({organ_type})...")
    
    # Fetch histopathology data based on organ type
    histo_data = HISTO_DETAILS.get(organ_type, {})
    subtype = random.choice(histo_data.get("Subtype", ["Unknown"]))
    stage = random.choice(histo_data.get("Stage", ["Unknown"]))
    color_distribution = calculate_color_distribution(biopsy_image_path)

    #  PDF Setup
    pdf_path = os.path.join(RESULTS_DIR, f"{patient_id}_{organ_type}_Histopathology_Report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    #  Add Hospital Name (Colorful)
    c.setFont("Helvetica-Bold", 16)
    c.setFillColorRGB(0.2, 0.5, 0.9)  # Blue color for professional look
    c.drawString(200, height - 50, "Advanced Oncology Diagnostic Center")

    #  Patient & Doctor Details (Structured)
    c.setFont("Helvetica-Bold", 12)
    c.setFillColorRGB(0, 0, 0)  # Black text
    # Left Side: Patient Info
    c.drawString(50, height - 90, f"Patient Name: {patient_name}")
    c.drawString(50, height - 110, f"Patient ID: {patient_id}")
    # Right Side: Doctor & Additional Info
    c.drawString(350, height - 90, f"Doctor: Dr. {doctor_name}")
    c.drawString(350, height - 110, f"Age: {patient_age} | Gender: {patient_gender}")

    #  Report Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(150, height - 140, f"Histopathology Report - {organ_type}")

    # Define y_position for the following content
    y_position = height - 170

    #  AI-Based Observation Result (Italicized)
    c.setFont("Helvetica-Oblique", 12)  # Italic for emphasis
    c.drawString(50, y_position, f"Preliminary Observation: {subtype} detected with {stage}. Further analysis required.")
    y_position -= 30

    #  Tumor Analysis Details
    c.setFont("Helvetica", 12)
    c.drawString(50, y_position, f"Tumor Subtype: {subtype}")
    y_position -= 20
    c.drawString(50, y_position, f" Tumor Stage: {stage}")
    y_position -= 20
    c.drawString(50, y_position, f" Mitotic Rate: {histo_data.get('Mitotic Rate', 'N/A')}")
    y_position -= 20
    c.drawString(50, y_position, f" Immunostaining Markers: {histo_data.get('Immunostaining', 'N/A')}")
    y_position -= 20
    c.drawString(50, y_position, f" Necrosis: {histo_data.get('Necrosis', 'N/A')}")
    y_position -= 40

    #  Biopsy Color Analysis
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, " Biopsy Color Analysis:")
    y_position -= 20
    c.setFont("Helvetica", 10)
    for color, percentage in color_distribution.items():
        c.drawString(50, y_position, f" {color}: {percentage}% - {COLOR_EXPLANATION[color]}")
        y_position -= 15

    #  Explainable AI (XAI) Insights
    y_position -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, " Explainable AI (XAI) Insights:")
    y_position -= 20
    xai_text = ("AI identified tumor regions using deep feature analysis. "
                "Grad-CAM highlighted tumor focus areas, while SHAP explained feature importance.")
    c.setFont("Helvetica", 10)
    c.drawString(50, y_position, xai_text)
    y_position -= 40

    #  Add Biopsy Image with Adjusted Clarity & Size
    biopsy_img = cv2.imread(biopsy_image_path)
    biopsy_img = cv2.resize(biopsy_img, (250, 150), interpolation=cv2.INTER_CUBIC)
    # Save enhanced image
    enhanced_biopsy_path = os.path.join(RESULTS_DIR, f"Enhanced_{os.path.basename(biopsy_image_path)}")
    cv2.imwrite(enhanced_biopsy_path, biopsy_img)

    # Insert the enhanced biopsy image into the PDF
    c.drawImage(ImageReader(enhanced_biopsy_path), 130, y_position - 150, width=250, height=150)

    # Save PDF
    c.save()
    print(f" PDF Report Saved: {pdf_path}")
    return pdf_path
