# Copyright (c) 2026 Khush Patil. All Rights Reserved.
# This code is for personal/educational viewing only. 
# Unauthorized copying, modification, or distribution is strictly prohibited.

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import io
import base64
import timm

app = FastAPI()

#API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define Classes and Maps ---
dev_classes = ['character_10_yna', 'character_11_taamatar', 'character_12_thaa', 'character_13_daa', 'character_14_dhaa', 'character_15_adna', 'character_16_tabala', 'character_17_tha', 'character_18_da', 'character_19_dha', 'character_1_ka', 'character_20_na', 'character_21_pa', 'character_22_pha', 'character_23_ba', 'character_24_bha', 'character_25_ma', 'character_26_yaw', 'character_27_ra', 'character_28_la', 'character_29_waw', 'character_2_kha', 'character_30_motosaw', 'character_31_petchiryakha', 'character_32_patalosaw', 'character_33_ha', 'character_34_chhya', 'character_35_tra', 'character_36_gya', 'character_3_ga', 'character_4_gha', 'character_5_kna', 'character_6_cha', 'character_7_chha', 'character_8_ja', 'character_9_jha', 'digit_0', 'digit_1', 'digit_2', 'digit_3', 'digit_4', 'digit_5', 'digit_6', 'digit_7', 'digit_8', 'digit_9']
dev_map = {'character_1_ka': 'क', 'character_2_kha': 'ख', 'character_3_ga': 'ग', 'character_4_gha': 'घ', 'character_5_kna': 'ङ', 'character_6_cha': 'च', 'character_7_chha': 'छ', 'character_8_ja': 'ज', 'character_9_jha': 'झ', 'character_10_yna': 'ञ', 'character_11_taamatar': 'ट', 'character_12_thaa': 'ठ', 'character_13_daa': 'ड', 'character_14_dhaa': 'ढ', 'character_15_adna': 'ण', 'character_16_tabala': 'त', 'character_17_tha': 'थ', 'character_18_da': 'द', 'character_19_dha': 'ध', 'character_20_na': 'न', 'character_21_pa': 'प', 'character_22_pha': 'फ', 'character_23_ba': 'ब', 'character_24_bha': 'भ', 'character_25_ma': 'म', 'character_26_yaw': 'य', 'character_27_ra': 'र', 'character_28_la': 'ल', 'character_29_waw': 'व', 'character_30_motosaw': 'श', 'character_31_petchiryakha': 'ष', 'character_32_patalosaw': 'स', 'character_33_ha': 'ह', 'character_34_chhya': 'क्ष', 'character_35_tra': 'त्र', 'character_36_gya': 'ज्ञ', 'digit_0': '०', 'digit_1': '१', 'digit_2': '२', 'digit_3': '३', 'digit_4': '४', 'digit_5': '५', 'digit_6': '६', 'digit_7': '७', 'digit_8': '८', 'digit_9': '९'}
emnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

sans_classes = ['bra', 'dda', 'dma', 'dra', 'dwa', 'dya', 'gri', 'hra', 'hri', 'hya', 'kra', 'mra', 'mri', 'pra', 'rhi', 'shra', 'tra', 'traw', 'tta']
sans_map = {'bra': 'ब्र', 'dda': 'ड्ड', 'dma': 'द्म', 'dra': 'द्र', 'dwa': 'द्व', 'dya': 'द्य', 'gri': 'गृ', 'hra': 'ह्र', 'hri': 'हृ', 'hya': 'ह्य', 'kra': 'क्र', 'mra': 'म्र', 'mri': 'मृ', 'pra': 'प्र', 'rhi': 'ऋ',    'shra': 'श्र', 'tra': 'त्र', 'traw': 'त्रौ','tta': 'ट्ट'}

# --- Define Architectures ---
class SimpleCNN(nn.Module):
    def __init__(self, out_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 56 * 56, 128), nn.ReLU(), nn.Linear(128, out_classes))
    def forward(self, x): return self.classifier(self.features(x))

class HybridModel(nn.Module):
    def __init__(self, out_classes):
        super(HybridModel, self).__init__()
        resnet = timm.create_model('resnet18', pretrained=False) 
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(512, out_classes)
    def forward(self, x):
        x = self.cnn_backbone(x)
        x = x.flatten(2).permute(0, 2, 1)
        return self.fc(self.transformer(x).mean(dim=1))

# --- Load Models ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = {'dev': {}, 'emnist': {}, 'sans': {}}

print("Loading models into API...")
try:
    # DEVANAGARI
    models['dev']['cnn'] = SimpleCNN(46).to(device)
    models['dev']['cnn'].load_state_dict(torch.load('Models/Devanagari/cnn_dev.pth', map_location=device))
    models['dev']['vit'] = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=46).to(device)
    models['dev']['vit'].load_state_dict(torch.load('Models/Devanagari/vit_dev.pth', map_location=device))
    models['dev']['hybrid'] = HybridModel(46).to(device)
    models['dev']['hybrid'].load_state_dict(torch.load('Models/Devanagari/hybrid_dev.pth', map_location=device))

    # EMNIST
    models['emnist']['cnn'] = SimpleCNN(47).to(device)
    models['emnist']['cnn'].load_state_dict(torch.load('Models/EMNIST/cnn_emnist.pth', map_location=device))
    models['emnist']['vit'] = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=47).to(device)
    models['emnist']['vit'].load_state_dict(torch.load('Models/EMNIST/vit_emnist.pth', map_location=device))
    models['emnist']['hybrid'] = HybridModel(47).to(device)
    models['emnist']['hybrid'].load_state_dict(torch.load('Models/EMNIST/hybrid_emnist.pth', map_location=device))

    # --- SANSKRIT ADDED ---
    num_sans = len(sans_classes)
    models['sans']['cnn'] = SimpleCNN(num_sans).to(device)
    models['sans']['cnn'].load_state_dict(torch.load('Models/Sanskrit/cnn_sans.pth', map_location=device))
    models['sans']['vit'] = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_sans).to(device)
    models['sans']['vit'].load_state_dict(torch.load('Models/Sanskrit/vit_sans.pth', map_location=device))
    models['sans']['hybrid'] = HybridModel(num_sans).to(device)
    models['sans']['hybrid'].load_state_dict(torch.load('Models/Sanskrit/hybrid_sans.pth', map_location=device))

    for dataset in models:
        for model in models[dataset].values():
            model.eval()
    print("API Ready!")
except Exception as e:
    print(f"Error loading models: {e}\nPlease check your 'Models' directory paths.")


# --- The Prediction Endpoint ---
@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    image_data = data['image'].split(',')[1] # Remove the base64 prefix
    dataset_mode = data['dataset_mode'] # 'dev', 'emnist', or 'sans'

    # Convert Base64 back to PIL Image
    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes))
    
    # --- 1. BASE THRESHOLDING ---
    img = img.convert('L')
    threshold = 140 
    img = img.point(lambda p: 255 if p > threshold else 0)

    # --- 2. AUTO-CROPPING & SQUARING ---
    # Temporarily invert to find the bounding box of the black ink
    bbox = ImageOps.invert(img).getbbox()
    
    if bbox:
        # Crop tightly around the character
        img = img.crop(bbox)
        
        # Pad the image to make it a perfect square (prevents stretching)
        max_size = max(img.size)
        delta_w = max_size - img.size[0]
        delta_h = max_size - img.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        
        # VERY IMPORTANT: Fill with 255 (White) so the background stays clean
        img = ImageOps.expand(img, border=padding, fill=255)
        
        # Add a standard 20-pixel white margin
        img = ImageOps.expand(img, border=20, fill=255)

    img = img.convert('RGB')
    
    # --- 3. CONDITIONAL INVERSION ---
    # 👉 Only invert colors for Devanagari and EMNIST
    if dataset_mode in ['dev', 'emnist']:
        img = ImageOps.invert(img)
    # Sanskrit skips the inversion to keep the white background / black text!

    # --- 4. TENSOR TRANSFORMATION ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0).to(device)

    # --- 5. INFERENCE LOGIC ---
    def get_pred(model, tensor, mode):
        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)
            top_p, top_class = probs.topk(1, dim=1)
            
            if mode == "dev":
                raw_name = dev_classes[top_class.item()]
                script = dev_map.get(raw_name, raw_name)
                return f"{script} ({raw_name.split('_')[-1].capitalize()})", round(top_p.item() * 100, 2)
            
            elif mode == "sans":
                raw_name = sans_classes[top_class.item()]
                script = sans_map.get(raw_name, raw_name)
                return f"{script} ({raw_name.capitalize()})", round(top_p.item() * 100, 2)
            
            else:
                return emnist_classes[top_class.item()], round(top_p.item() * 100, 2)

    cnn_pred, cnn_conf = get_pred(models[dataset_mode]['cnn'], tensor, dataset_mode)
    vit_pred, vit_conf = get_pred(models[dataset_mode]['vit'], tensor, dataset_mode)
    hyb_pred, hyb_conf = get_pred(models[dataset_mode]['hybrid'], tensor, dataset_mode)

    return {
        "cnn": {"label": cnn_pred, "confidence": cnn_conf},
        "vit": {"label": vit_pred, "confidence": vit_conf},
        "hybrid": {"label": hyb_pred, "confidence": hyb_conf}
    }