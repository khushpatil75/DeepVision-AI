# Copyright (c) 2026 Khush Patil. All Rights Reserved.
# This code is for personal/educational viewing only. 
# Unauthorized copying, modification, or distribution is strictly prohibited.
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import timm

# --- 1. Page Config & Custom CSS (The Aesthetic Upgrade) ---
st.set_page_config(page_title="DeepVision | Character Recognition", layout="wide", initial_sidebar_state="expanded")

# Injecting CSS to hide default Streamlit menus, adjust padding, and style fonts
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stProgress .st-bo {
        background-color: #4CAF50; /* Sleek green progress bar */
    }
</style>
""", unsafe_allow_html=True)

st.title("DeepVision AI")
st.markdown("#### Universal Handwritten Character Recognition System")
st.divider()

# --- 2. Class Definitions & Maps ---
dev_classes = [
    'character_10_yna', 'character_11_taamatar', 'character_12_thaa', 'character_13_daa', 'character_14_dhaa', 
    'character_15_adna', 'character_16_tabala', 'character_17_tha', 'character_18_da', 'character_19_dha', 
    'character_1_ka', 'character_20_na', 'character_21_pa', 'character_22_pha', 'character_23_ba', 
    'character_24_bha', 'character_25_ma', 'character_26_yaw', 'character_27_ra', 'character_28_la', 
    'character_29_waw', 'character_2_kha', 'character_30_motosaw', 'character_31_petchiryakha', 
    'character_32_patalosaw', 'character_33_ha', 'character_34_chhya', 'character_35_tra', 'character_36_gya', 
    'character_3_ga', 'character_4_gha', 'character_5_kna', 'character_6_cha', 'character_7_chha', 
    'character_8_ja', 'character_9_jha', 'digit_0', 'digit_1', 'digit_2', 'digit_3', 'digit_4', 
    'digit_5', 'digit_6', 'digit_7', 'digit_8', 'digit_9'
]

dev_map = {
    'character_1_ka': 'क', 'character_2_kha': 'ख', 'character_3_ga': 'ग', 'character_4_gha': 'घ', 'character_5_kna': 'ङ',
    'character_6_cha': 'च', 'character_7_chha': 'छ', 'character_8_ja': 'ज', 'character_9_jha': 'झ', 'character_10_yna': 'ञ',
    'character_11_taamatar': 'ट', 'character_12_thaa': 'ठ', 'character_13_daa': 'ड', 'character_14_dhaa': 'ढ', 'character_15_adna': 'ण',
    'character_16_tabala': 'त', 'character_17_tha': 'थ', 'character_18_da': 'द', 'character_19_dha': 'ध', 'character_20_na': 'न',
    'character_21_pa': 'प', 'character_22_pha': 'फ', 'character_23_ba': 'ब', 'character_24_bha': 'भ', 'character_25_ma': 'म',
    'character_26_yaw': 'य', 'character_27_ra': 'र', 'character_28_la': 'ल', 'character_29_waw': 'व', 'character_30_motosaw': 'श',
    'character_31_petchiryakha': 'ष', 'character_32_patalosaw': 'स', 'character_33_ha': 'ह', 'character_34_chhya': 'क्ष',
    'character_35_tra': 'त्र', 'character_36_gya': 'ज्ञ',
    'digit_0': '०', 'digit_1': '१', 'digit_2': '२', 'digit_3': '३', 'digit_4': '४', 
    'digit_5': '५', 'digit_6': '६', 'digit_7': '७', 'digit_8': '८', 'digit_9': '९'
}

emnist_classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

# --- 3. Dynamic Model Architectures ---
class SimpleCNN(nn.Module):
    def __init__(self, out_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128), nn.ReLU(),
            nn.Linear(128, out_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

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
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# --- 4. Load Models into Memory ---
@st.cache_resource
def load_all_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_dict = {'dev': {}, 'emnist': {}}
    
    try:
        # Load Devanagari (46 classes)
        dev_cnn = SimpleCNN(46).to(device)
        dev_cnn.load_state_dict(torch.load('Models/Devanagari/cnn_dev.pth', map_location=device))
        dev_vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=46).to(device)
        dev_vit.load_state_dict(torch.load('Models/Devanagari/vit_dev.pth', map_location=device))
        dev_hyb = HybridModel(46).to(device)
        dev_hyb.load_state_dict(torch.load('Models/Devanagari/hybrid_dev.pth', map_location=device))
        
        models_dict['dev'] = {'cnn': dev_cnn.eval(), 'vit': dev_vit.eval(), 'hybrid': dev_hyb.eval()}
        
        # Load EMNIST (47 classes)
        emnist_cnn = SimpleCNN(47).to(device)
        emnist_cnn.load_state_dict(torch.load('Models/EMNIST/cnn_emnist.pth', map_location=device))
        emnist_vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=47).to(device)
        emnist_vit.load_state_dict(torch.load('Models/EMNIST/vit_emnist.pth', map_location=device))
        emnist_hyb = HybridModel(47).to(device)
        emnist_hyb.load_state_dict(torch.load('Models/EMNIST/hybrid_emnist.pth', map_location=device))
        
        models_dict['emnist'] = {'cnn': emnist_cnn.eval(), 'vit': emnist_vit.eval(), 'hybrid': emnist_hyb.eval()}
        
    except FileNotFoundError as e:
        st.error(f"Missing a model file! Ensure all 6 .pth files are in your 'models' folder. Error: {e}")
        st.stop()
        
    return models_dict, device

all_models, device = load_all_models()

# --- 5. Sidebar Layout ---
with st.sidebar:
    st.markdown("### System Configuration")
    app_mode = st.radio("Active Engine:", ["Devanagari (Hindi)", "English (EMNIST)"])
    st.divider()
    
    st.markdown("### About the Project")
    st.info("""
    **Architecture:** Parallel Inference
    **Models:** CNN, ViT, Hybrid (CNN+Transformer)
    **Pipeline:** PyTorch -> torchvision -> Streamlit
    """)

if app_mode == "Devanagari (Hindi)":
    current_models = all_models['dev']
    current_classes = dev_classes
    st.markdown("### Active Engine: **Devanagari Script**")
else:
    current_models = all_models['emnist']
    current_classes = emnist_classes
    st.markdown("### Active Engine: **English (Alphabets & Digits)**")

# --- 6. Image Processing ---
def process_image(img_array=None, pil_img=None):
    if img_array is not None:
        img = Image.fromarray(img_array.astype('uint8'), 'RGBA').convert('RGB')
    else:
        img = pil_img.convert('RGB')
        
    img = ImageOps.invert(img)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(device)

def get_prediction(model, tensor, mode):
    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1)
        top_p, top_class = probabilities.topk(1, dim=1)
        
        if mode == "Devanagari (Hindi)":
            raw_name = current_classes[top_class.item()]
            actual_script = dev_map.get(raw_name, raw_name)
            english_name = raw_name.split('_')[-1].capitalize()
            display_text = f"{actual_script} ({english_name})"
        else:
            display_text = current_classes[top_class.item()]
            
        return display_text, (top_p.item() * 100)

# --- 7. Main UI Layout ---
col1, col2, col3 = st.columns([1, 0.1, 1.2]) # Added a spacer column for cleaner look

with col1:
    st.markdown("##### 1. Input Module")
    # Clean styling for the canvas
    canvas_result = st_canvas(
        fill_color="#FFFFFF", stroke_width=20, stroke_color="#000000",
        background_color="#FFFFFF", height=280, width=280,
        drawing_mode="freedraw", key="canvas",
    )
    
    uploaded_file = st.file_uploader("Or upload an image (Black text on white)", type=["jpg", "jpeg", "png"])

with col3:
    st.markdown("##### 2. Inference Engine Output")
    input_tensor = None
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Processed Upload', width=100)
        input_tensor = process_image(pil_img=image)
        
    elif canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
        input_tensor = process_image(img_array=canvas_result.image_data)

    if input_tensor is not None:
        with st.spinner('Running parallel inference...'):
            cnn_pred, cnn_conf = get_prediction(current_models['cnn'], input_tensor, app_mode)
            vit_pred, vit_conf = get_prediction(current_models['vit'], input_tensor, app_mode)
            hyb_pred, hyb_conf = get_prediction(current_models['hybrid'], input_tensor, app_mode)
            
            # Premium Dashboard Metric Layout
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.metric(label="CNN Model", value=cnn_pred)
                st.progress(int(cnn_conf), text=f"{cnn_conf:.1f}% Conf")
                
            with m2:
                st.metric(label="ViT Model", value=vit_pred)
                st.progress(int(vit_conf), text=f"{vit_conf:.1f}% Conf")
                
            with m3:
                st.metric(label="Hybrid Model", value=hyb_pred)
                st.progress(int(hyb_conf), text=f"{hyb_conf:.1f}% Conf")
    else:
        st.info("Awaiting input from canvas or image upload...")