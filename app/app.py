"""
DentalVision AI: Automated Dental Caries and Plaque Detection Diagnostic Suite
Built with Streamlit and PyTorch UNet (ResNet-34)
"""

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
import segmentation_models_pytorch as smp

# --- Page Configuration ---
st.set_page_config(
    page_title="DentalVision AI | Caries & Plaque Detection",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path Configurations ---
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
CARIES_MODEL_PATH = MODELS_DIR / "best_caries_unet.pth"
PLAQUE_MODEL_PATH = MODELS_DIR / "plaque_unet.pth"

PERIAPICAL_SAMPLE_DIR = ROOT_DIR / "Datasets" / "Intraoral Radiographic Database for Periapical Les" / "images"
PLAQUE_SAMPLE_DIR = ROOT_DIR / "Datasets" / "ODS_dataset" / "Plaques" / "img"

DEMO_DIR = ROOT_DIR / "demo testing"
DEMO_CARIES_IMG = DEMO_DIR / "caries" / "images"
DEMO_CARIES_GT = DEMO_DIR / "caries" / "ground_truth_reference"
DEMO_PLAQUE_IMG = DEMO_DIR / "plaque" / "images"
DEMO_PLAQUE_GT = DEMO_DIR / "plaque" / "ground_truth_reference"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Predefined Ground Truth Metadata for Curated Demo Cases
CARIES_DEMO_CASES = {
    "patient_0001.jpg": {
        "title": "patient_0001.jpg - Sound Tooth (Negative)",
        "category": "Sound / Healthy Enamel",
        "expected_count": 0,
        "expected_area": "0.00%",
        "notes": "No active caries. Contains 3 radio-opaque restorative fillings."
    },
    "patient_0002.jpg": {
        "title": "patient_0002.jpg - Sound Enamel (Caries-free)",
        "category": "Sound / Caries-free",
        "expected_count": 0,
        "expected_area": "0.00%",
        "notes": "Caries-free crowns. Note: Root apex exhibits a periapical lesion."
    },
    "patient_0004.jpg": {
        "title": "patient_0004.jpg - Single Cavity (Mild/Incipient)",
        "category": "Mild / Incipient Caries",
        "expected_count": 1,
        "expected_area": "2.16%",
        "notes": "Distinct coronal interproximal radiolucency on crown."
    },
    "patient_0006.jpg": {
        "title": "patient_0006.jpg - Single Cavity (Mild Enamel Decay)",
        "category": "Mild / Enamel Caries",
        "expected_count": 1,
        "expected_area": "1.53%",
        "notes": "Localized demineralization at the enamel-dentin junction."
    },
    "patient_0030.jpg": {
        "title": "patient_0030.jpg - Recurrent Caries (Adjacent to Restoration)",
        "category": "Mild / Recurrent Caries",
        "expected_count": 1,
        "expected_area": "1.65%",
        "notes": "Secondary decay adjacent to existing restoration margin."
    },
    "patient_0016.jpg": {
        "title": "patient_0016.jpg - Multiple Cavities (Moderate)",
        "category": "Moderate Caries (2 Lesions)",
        "expected_count": 2,
        "expected_area": "3.88%",
        "notes": "2 separate carious lesions involving proximal surfaces of adjacent teeth."
    },
    "patient_0023.jpg": {
        "title": "patient_0023.jpg - Deep Cavities (Moderate)",
        "category": "Moderate Dentinal Caries (2 Lesions)",
        "expected_count": 2,
        "expected_area": "4.12%",
        "notes": "2 deep cavities advancing into dentin towards the pulp chamber."
    },
    "patient_0044.jpg": {
        "title": "patient_0044.jpg - Severe Decay (3 Lesions)",
        "category": "Severe Multi-surface Caries",
        "expected_count": 3,
        "expected_area": "3.44%",
        "notes": "Widespread caries across multiple teeth with structural loss."
    },
    "patient_0055.png": {
        "title": "patient_0055.png - Extensive Deep Cavities (3 Lesions)",
        "category": "Severe Deep Decay",
        "expected_count": 3,
        "expected_area": "3.29%",
        "notes": "Extensive coronal destruction and deep dentinal cavitation."
    }
}

PLAQUE_DEMO_CASES = {
    "10.png": {
        "title": "10.png - Clean Teeth (Zero Plaque)",
        "category": "Clean / Sound Surface",
        "expected_count": 0,
        "expected_area": "0.00%",
        "notes": "Plaque-free smooth enamel surfaces."
    },
    "1004.png": {
        "title": "1004.png - Healthy Gingival Margins (Zero Plaque)",
        "category": "Clean / Sound Gingival Margin",
        "expected_count": 0,
        "expected_area": "0.00%",
        "notes": "Clean gingival margins without biofilm accumulation."
    },
    "1194.png": {
        "title": "1194.png - Incipient Plaque (< 1%)",
        "category": "Mild Plaque Accumulation",
        "expected_count": 1,
        "expected_area": "0.61%",
        "notes": "Localized incipient bacterial accumulation near the gingival border."
    },
    "1250.png": {
        "title": "1250.png - Localized Cervical Plaque (< 1%)",
        "category": "Mild Plaque Accumulation",
        "expected_count": 1,
        "expected_area": "0.64%",
        "notes": "Small cervical biofilm deposit along the gum line."
    },
    "1069.png": {
        "title": "1069.png - Moderate Interdental Plaque (~5%)",
        "category": "Moderate Plaque Biofilm",
        "expected_count": 2,
        "expected_area": "5.14%",
        "notes": "Plaque deposits extending across interproximal tooth surfaces."
    },
    "1014.png": {
        "title": "1014.png - Moderate Cervical Plaque (~6%)",
        "category": "Moderate Plaque Biofilm",
        "expected_count": 2,
        "expected_area": "6.25%",
        "notes": "Visible continuous biofilm along the cervical third of teeth."
    },
    "100.png": {
        "title": "100.png - Heavy Plaque Biofilm (~15%)",
        "category": "Heavy / Severe Plaque Biofilm",
        "expected_count": 2,
        "expected_area": "15.01%",
        "notes": "Extensive continuous biofilm coating multiple dental crowns."
    },
    "1040.png": {
        "title": "1040.png - Confluent Severe Plaque (~17%)",
        "category": "Heavy / Severe Plaque Biofilm",
        "expected_count": 3,
        "expected_area": "16.71%",
        "notes": "Thick confluent plaque deposits spanning multiple anterior/posterior teeth."
    }
}

# --- Custom Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .ground-truth-card {
        background-color: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1rem;
    }
    .severity-badge-none {
        background-color: #ecfdf5;
        color: #065f46;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .severity-badge-mild {
        background-color: #fef3c7;
        color: #92400e;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .severity-badge-severe {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# --- Model Loader ---
@st.cache_resource(show_spinner="Loading UNet Diagnostic Network...")
def load_unet_model(model_path_str: str, mtime: float):
    model_path = Path(model_path_str)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if not model_path.exists() else None,
        in_channels=3,
        classes=1,
        activation=None
    )
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        status = "Trained Checkpoint Loaded"
    else:
        status = "Base Encoder Loaded"
    
    model.to(DEVICE)
    model.eval()
    return model, status


# --- Image Processing Utilities ---
def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def run_inference(model, img_rgb: np.ndarray, target_size=(512, 512)):
    h_orig, w_orig = img_rgb.shape[:2]
    resized = cv2.resize(img_rgb, target_size)
    
    # ImageNet normalization
    norm_img = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_img = (norm_img - mean) / std
    
    tensor_input = torch.tensor(norm_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
    prob_orig = cv2.resize(probs, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return prob_orig


def postprocess_findings(prob_mask: np.ndarray, threshold: float = 0.4, min_area: int = 40):
    binary_mask = (prob_mask >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    findings = []
    filtered_mask = np.zeros_like(binary_mask)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            cv2.drawContours(filtered_mask, [cnt], -1, 1, thickness=-1)
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            cx = int(M["m10"] / (M["m00"] + 1e-6))
            cy = int(M["m01"] / (M["m00"] + 1e-6))
            findings.append({
                'contour': cnt,
                'bbox': (x, y, w, h),
                'area': int(area),
                'centroid': (cx, cy)
            })
            
    total_pixels = prob_mask.shape[0] * prob_mask.shape[1]
    affected_area_pct = (np.sum(filtered_mask) / total_pixels) * 100.0
    return findings, filtered_mask, affected_area_pct


# --- Sidebar Navigation & Controls ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/tooth.png", width=64)
    st.title("DentalVision AI")
    st.caption("Clinical Decision Support for Caries & Plaque")
    st.markdown("---")
    
    mode = st.radio(
        "Select Diagnostic Modality",
        ["🦷 Intraoral Radiograph (Caries / Decay)", "📷 Clinical Camera (Plaque / Biofilm)"],
        index=0
    )
    is_radiograph_mode = "Radiograph" in mode
    
    st.markdown("### ⚙️ Diagnostic Parameters")
    
    # Recommended default thresholds
    default_th = 0.40 if is_radiograph_mode else 0.40
    conf_threshold = st.slider(
        "Sensitivity / Detection Threshold",
        min_value=0.10, max_value=0.90, value=default_th, step=0.05,
        help="Optimal range: 0.35 - 0.45 for X-ray cavities, 0.35 - 0.50 for photographic plaque."
    )
    
    # CLAHE option only for monochrome radiographs
    if is_radiograph_mode:
        enable_clahe = st.checkbox(
            "Apply CLAHE Contrast Enhancement",
            value=True,
            help="Improves subtle demineralization boundaries on low-contrast X-rays."
        )
    else:
        enable_clahe = False
        st.caption("ℹ️ *Photographs use natural RGB color analysis. CLAHE is automatically disabled.*")
    
    overlay_opacity = st.slider(
        "Diagnostic Overlay Opacity",
        min_value=0.1, max_value=0.9, value=0.45, step=0.05
    )
    
    show_bboxes = st.checkbox("Show Bounding Boxes", value=True)
    
    st.markdown("---")
    st.markdown(f"**Compute Device:** `{DEVICE}`")
    if torch.cuda.is_available():
        st.caption(f"GPU: {torch.cuda.get_device_name(0)}")


# --- Main Application Header ---
st.markdown('<div class="main-header">Automated Dental Healthcare Diagnostic Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Learning Semantic Segmentation (UNet) for Tooth Decay & Plaque Detection</div>', unsafe_allow_html=True)

# Select active model and directory based on mode
active_model_path = CARIES_MODEL_PATH if is_radiograph_mode else PLAQUE_MODEL_PATH
demo_img_dir = DEMO_CARIES_IMG if is_radiograph_mode else DEMO_PLAQUE_IMG
demo_gt_dir = DEMO_CARIES_GT if is_radiograph_mode else DEMO_PLAQUE_GT
demo_meta = CARIES_DEMO_CASES if is_radiograph_mode else PLAQUE_DEMO_CASES

# Mode-Specific Clinical Terms
if is_radiograph_mode:
    target_label = "Dental Caries (Demineralized Cavities)"
    metric_count_label = "Carious Lesions Detected"
    metric_area_label = "Caries Lesion Area"
    metric_prob_label = "Max Caries Probability"
    table_item_label = "Lesion #"
    table_header = "#### 3. Localized Caries Lesion Quantification"
    mask_color = (255, 30, 30) # Red
else:
    target_label = "Bacterial Plaque & Biofilm Deposit"
    metric_count_label = "Biofilm Regions Detected"
    metric_area_label = "Plaque Biofilm Coverage"
    metric_prob_label = "Max Biofilm Probability"
    table_item_label = "Biofilm Region #"
    table_header = "#### 3. Localized Plaque Biofilm Quantification"
    mask_color = (255, 140, 0) # Orange

mtime = active_model_path.stat().st_mtime if active_model_path.exists() else 0.0
model, model_status = load_unet_model(str(active_model_path), mtime)

if not active_model_path.exists():
    st.info(f"ℹ️ **Notice:** `{active_model_path.name}` not found in `models/`. Running with base encoder weights. Run training in `Notebooks/Main.ipynb` to save trained weights.")

# --- Image Input Selection ---
st.markdown("#### 1. Input Dental Image")
input_option = st.radio(
    "Choose Input Method:", 
    ["🎯 Curated Demo Testing Suite (with Expected Results)", "📁 Upload Image File", "🗂️ Browse Full Dataset"], 
    horizontal=True
)

input_image = None
image_name = "Sample"
selected_meta = None
ref_image_path = None

if input_option == "🎯 Curated Demo Testing Suite (with Expected Results)":
    case_keys = list(demo_meta.keys())
    case_labels = [demo_meta[k]["title"] for k in case_keys]
    chosen_label = st.selectbox("Select a Pre-curated Clinical Test Case:", case_labels)
    chosen_idx = case_labels.index(chosen_label)
    chosen_key = case_keys[chosen_idx]
    
    selected_meta = demo_meta[chosen_key]
    img_path = demo_img_dir / chosen_key
    if img_path.exists():
        bgr = cv2.imread(str(img_path))
        input_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image_name = chosen_key
        
        ref_stem = Path(chosen_key).stem
        candidate_ref = demo_gt_dir / f"{ref_stem}_reference.png"
        if candidate_ref.exists():
            ref_image_path = candidate_ref

    # Ground Truth Expected Results Card
    target_unit = "Lesions" if is_radiograph_mode else "Biofilm Regions"
    st.markdown(f"""
    <div class="ground-truth-card">
        <strong>📋 Clinical Ground Truth for <code>{image_name}</code>:</strong><br>
        • <strong>Category:</strong> {selected_meta['category']}<br>
        • <strong>Expected {target_unit}:</strong> {selected_meta.get('expected_count', 'N/A')}<br>
        • <strong>Expected Coverage:</strong> {selected_meta['expected_area']}<br>
        • <strong>Clinical Landmark / Notes:</strong> {selected_meta['notes']}
    </div>
    """, unsafe_allow_html=True)

elif input_option == "📁 Upload Image File":
    source_subfolder = "caries" if is_radiograph_mode else "plaque"
    st.caption(f"Drag and drop any image from `demo testing/{source_subfolder}/images/` below:")
    uploaded_file = st.file_uploader(
        "Upload Dental Image (JPG, PNG, JPEG)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        input_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image_name = uploaded_file.name

else: # Browse Full Dataset
    sample_dir = PERIAPICAL_SAMPLE_DIR if is_radiograph_mode else PLAQUE_SAMPLE_DIR
    if sample_dir.exists():
        sample_files = sorted([f.name for f in sample_dir.glob("*.*") if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])[:30]
        selected_file = st.selectbox("Choose a sample from full dataset:", sample_files)
        if selected_file:
            img_path = sample_dir / selected_file
            bgr = cv2.imread(str(img_path))
            if bgr is not None:
                input_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image_name = selected_file
    else:
        st.warning(f"Dataset directory not found: {sample_dir}")

# --- Diagnostic Inference & Display ---
if input_image is not None:
    st.markdown("---")
    st.markdown("#### 2. Clinical Diagnostic Analysis")
    
    # Preprocessing
    processed_image = apply_clahe(input_image) if enable_clahe else input_image.copy()
    
    # Run model
    prob_mask = run_inference(model, processed_image)
    findings, filtered_mask, affected_pct = postprocess_findings(prob_mask, threshold=conf_threshold)
    
    # Create Visual Overlay
    overlay_mask = np.zeros_like(input_image)
    overlay_mask[filtered_mask == 1] = mask_color
    blended = cv2.addWeighted(input_image, 1.0 - overlay_opacity, overlay_mask, overlay_opacity, 0)
    
    # Draw contours & bounding boxes
    for idx, finding in enumerate(findings):
        cnt = finding['contour']
        contour_color = (255, 255, 0) if is_radiograph_mode else (0, 255, 0)
        cv2.drawContours(blended, [cnt], -1, contour_color, 2)
        if show_bboxes:
            x, y, w, h = finding['bbox']
            bbox_color = (0, 255, 255)
            cv2.rectangle(blended, (x, y), (x + w, y + h), bbox_color, 2)
            cv2.putText(blended, f"#{idx+1}", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)
    
    # Top Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label=metric_count_label, value=len(findings))
    with col2:
        st.metric(label=metric_area_label, value=f"{affected_pct:.2f}%")
    with col3:
        st.metric(label=metric_prob_label, value=f"{np.max(prob_mask):.2f}")
    with col4:
        if is_radiograph_mode:
            # Caries Diagnostic Assessment
            if len(findings) == 0:
                st.markdown('**Severity Assessment:**<br><span class="severity-badge-none">Sound / No Active Decay</span>', unsafe_allow_html=True)
            elif affected_pct < 1.5:
                st.markdown('**Severity Assessment:**<br><span class="severity-badge-mild">Incipient / Enamel Caries</span>', unsafe_allow_html=True)
            else:
                st.markdown('**Severity Assessment:**<br><span class="severity-badge-severe">Moderate - Deep Dentinal Caries</span>', unsafe_allow_html=True)
        else:
            # Plaque Oral Hygiene Assessment
            if len(findings) == 0:
                st.markdown('**Oral Hygiene Assessment:**<br><span class="severity-badge-none">Clean Surfaces (No Biofilm)</span>', unsafe_allow_html=True)
            elif affected_pct < 5.0:
                st.markdown('**Oral Hygiene Assessment:**<br><span class="severity-badge-mild">Mild Plaque Accumulation</span>', unsafe_allow_html=True)
            else:
                st.markdown('**Oral Hygiene Assessment:**<br><span class="severity-badge-severe">Moderate - Heavy Biofilm Coverage</span>', unsafe_allow_html=True)

    st.write("")
    
    # Visual Columns
    img_col1, img_col2, img_col3 = st.columns(3)
    with img_col1:
        st.image(input_image, caption=f"Raw Image ({image_name})", use_container_width=True)
    with img_col2:
        second_caption = "CLAHE Contrast Enhanced" if enable_clahe else "Full-Color RGB Camera View"
        st.image(processed_image, caption=second_caption, use_container_width=True)
    with img_col3:
        st.image(blended, caption=f"Diagnostic Overlay ({target_label})", use_container_width=True)
        
    # Localized Findings Quantification Table
    if len(findings) > 0:
        st.markdown(table_header)
        table_data = []
        for i, f_item in enumerate(findings):
            table_data.append({
                table_item_label: f"#{i+1}",
                "Centroid (X, Y)": f"({f_item['centroid'][0]}, {f_item['centroid'][1]})",
                "Bounding Box (X, Y, W, H)": f"{f_item['bbox']}",
                "Area (pixels)": f_item['area'],
                "Coverage % of Image": f"{(f_item['area'] / (input_image.shape[0] * input_image.shape[1])) * 100:.3f}%"
            })
        st.dataframe(table_data, use_container_width=True)
        
    # Expandable Ground Truth Reference Viewer
    if ref_image_path is not None and ref_image_path.exists():
        ref_title = "Radiologist Ground-Truth Annotation" if is_radiograph_mode else "Clinical Plaque Ground-Truth Annotation"
        with st.expander(f"🔍 View {ref_title} Reference (Side-by-Side)", expanded=False):
            ref_bgr = cv2.imread(str(ref_image_path))
            if ref_bgr is not None:
                ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
                st.image(ref_rgb, caption=f"Ground Truth Reference: {ref_image_path.name} (Left: Input Image, Right: Ground Truth Mask)", use_container_width=True)
else:
    st.info("👆 Please upload an intraoral image or select a pre-curated demo case above to initiate diagnostic analysis.")
