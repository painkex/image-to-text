import streamlit as st
import cv2
import torch
import time
import warnings
import os
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Florence-2 Vision AI",
    layout="centered"
)

st.title("👁️ IA de Vision en Temps Réel")
st.markdown("Analyse en direct avec Florence-2")

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERIFY_SCHEDULED_REMOVAL"] = "false"

MODEL_ID = "microsoft/Florence-2-base"
DEVICE = "cpu"


# --------------------------------------------------
# LOAD MODEL (CACHE)
# --------------------------------------------------

@st.cache_resource
def load_model():

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    ).to(DEVICE)

    # optimisation stabilité
    model.config.use_cache = False

    return processor, model


processor, model = load_model()


# --------------------------------------------------
# UI
# --------------------------------------------------

col1, col2 = st.columns([2, 1])

with col1:
    run = st.checkbox("Activer la caméra")

    frame_placeholder = st.empty()

with col2:
    st.subheader("Résultat")

    description_box = st.empty()

    objects_box = st.empty()


# --------------------------------------------------
# CAMERA LOOP
# --------------------------------------------------

if run:

    cap = cv2.VideoCapture(0)

    last_analysis = 0

    ANALYSIS_INTERVAL = 5  # secondes


    while run:

        ret, frame = cap.read()

        if not ret:
            st.error("Impossible d'accéder à la caméra")
            break


        # afficher image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(
            frame_rgb,
            channels="RGB",
            use_container_width=True
        )


        # analyse périodique
        current_time = time.time()

        if current_time - last_analysis > ANALYSIS_INTERVAL:

            last_analysis = current_time

            image = Image.fromarray(frame_rgb)


            # -------- description --------
            inputs_desc = processor(
                text="<MORE_DETAILED_CAPTION>",
                images=image,
                return_tensors="pt"
            ).to(DEVICE)


            # -------- objets --------
            inputs_od = processor(
                text="<OD>",
                images=image,
                return_tensors="pt"
            ).to(DEVICE)


            with torch.no_grad():

                # description
                output_desc = model.generate(
                    **inputs_desc,
                    max_new_tokens=150,
                    num_beams=1
                )

                description = processor.batch_decode(
                    output_desc,
                    skip_special_tokens=True
                )[0]


                # objets
                output_od = model.generate(
                    **inputs_od,
                    max_new_tokens=100,
                    num_beams=1
                )

                decoded_od = processor.batch_decode(
                    output_od,
                    skip_special_tokens=True
                )[0]


                parsed = processor.post_process_generation(
                    decoded_od,
                    task="<OD>",
                    image_size=(frame.shape[1], frame.shape[0])
                )


            # afficher description
            description_box.markdown(
                f"**📝 Description :**\n{description}"
            )


            # afficher objets
            if "labels" in parsed and parsed["labels"]:

                objects = list(set(parsed["labels"]))

                objects_box.markdown(
                    "**📦 Objets détectés :**\n" +
                    ", ".join(objects)
                )


        time.sleep(0.01)


    cap.release()


else:

    st.info("Activez la caméra pour démarrer.")