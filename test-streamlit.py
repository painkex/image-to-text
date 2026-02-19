import streamlit as st
import cv2
import torch
import time
import os
import warnings
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from PIL import Image
import numpy as np

# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="Florence-2 Vision AI", layout="centered")
st.title("👁️ IA de Vision en Temps Réel")
st.markdown("Cette application utilise **Florence-2** pour analyser votre environnement.")


# --- CACHE DU MODÈLE (Pour éviter de le recharger à chaque clic) ---
@st.cache_resource
def load_model():
    warnings.filterwarnings("ignore")
    os.environ["TRANSFORMERS_VERIFY_SCHEDULED_REMOVAL"] = "false"
    device = "cpu"
    model_id = "microsoft/Florence-2-base"

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config.use_cache = False
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, trust_remote_code=True, attn_implementation="eager"
    ).to(device)
    return processor, model, device


processor, model, device = load_model()

# --- INTERFACE ---
col1, col2 = st.columns([2, 1])
with col1:
    run_app = st.checkbox("Activer la Caméra", value=False)
    frame_placeholder = st.empty()  # Zone pour la vidéo

with col2:
    st.subheader("Détails de l'analyse")
    status_text = st.empty()
    objects_text = st.empty()

# --- LOGIQUE DE CAPTURE ---
if run_app:
    cap = cv2.VideoCapture(0)
    last_analysis_time = 0
    analysis_frequency = 5  # Secondes

    while run_app:
        ret, frame = cap.read()
        if not ret:
            st.error("Impossible d'accéder à la webcam.")
            break

        # Conversion pour affichage Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        current_time = time.time()
        if current_time - last_analysis_time > analysis_frequency:
            last_analysis_time = current_time

            # Analyse IA
            image_pil = Image.fromarray(frame_rgb)

            # 1. Description
            inputs_desc = processor(text="<MORE_DETAILED_CAPTION>", images=image_pil, return_tensors="pt").to(device)
            # 2. Objets
            inputs_od = processor(text="<OD>", images=image_pil, return_tensors="pt").to(device)

            with torch.no_grad():
                # On génère
                out_desc = model.generate(**inputs_desc, max_new_tokens=150, num_beams=1, use_cache=False)
                description = processor.batch_decode(out_desc, skip_special_tokens=True)[0]

                out_od = model.generate(**inputs_od, max_new_tokens=100, num_beams=1, use_cache=False)
                pred_od = processor.batch_decode(out_od, skip_special_tokens=True)[0]
                parsed_od = processor.post_process_generation(pred_od, task="<OD>",
                                                              image_size=(frame.shape[1], frame.shape[0]))

            # Mise à jour de l'interface
            status_text.markdown(f"**📝 Description :**\n{description}")
            if "labels" in parsed_od:
                obj_list = ", ".join(set(parsed_od["labels"]))
                objects_text.markdown(f"**📦 Objets détectés :**\n{obj_list}")

        # Pour éviter de saturer le CPU Streamlit
        time.sleep(0.01)

    cap.release()
else:
    st.info("Cochez la case 'Activer la Caméra' pour commencer.")