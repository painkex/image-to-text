import cv2
import time
import torch
import os
import warnings
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from PIL import Image

# --- CONFIGURATION SYSTÈME ---
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERIFY_SCHEDULED_REMOVAL"] = "false"

device = "cpu"
model_id = "microsoft/Florence-2-base"

print(f"--- SYSTÈME D'ANALYSE HAUTE PERFORMANCE (TERMINAL ONLY) ---")

try:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config.use_cache = False
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, trust_remote_code=True, attn_implementation="eager"
    ).to(device)
    print("✅ Modèle chargé. Prêt pour l'analyse profonde.")
except Exception as e:
    print(f"❌ Erreur : {e}")
    exit()

# --- PARAMÈTRES ---
cap = cv2.VideoCapture(0)
analysis_frequency = 6  # On peut baisser à 6s car on ne consomme plus de GPU pour l'affichage
last_analysis_time = 0

print("\n🚀 Lancement. Surveillez le terminal ci-dessous pour les descriptions détaillées.")
print("Appuyez sur 'q' sur la fenêtre vidéo pour arrêter.\n")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Affichage simple du flux (léger)
    cv2.imshow('Camera (Analyse en cours dans le terminal)', frame)

    current_time = time.time()
    if current_time - last_analysis_time > analysis_frequency:
        last_analysis_time = current_time

        try:
            # Préparation image
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # --- ANALYSE DU CONTEXTE GLOBAL ---
            inputs_desc = processor(text="<MORE_DETAILED_CAPTION>", images=image_pil, return_tensors="pt").to(device)

            # --- ANALYSE DES OBJETS (L'ARÇON, OBJETS EN MAIN, ETC.) ---
            inputs_od = processor(text="<OD>", images=image_pil, return_tensors="pt").to(device)

            with torch.no_grad():
                # Génération de la description riche
                out_desc = model.generate(**inputs_desc, max_new_tokens=200, num_beams=1, use_cache=False)
                description = processor.batch_decode(out_desc, skip_special_tokens=True)[0]

                # Génération de la liste d'objets
                out_od = model.generate(**inputs_od, max_new_tokens=100, num_beams=1, use_cache=False)
                pred_od = processor.batch_decode(out_od, skip_special_tokens=True)[0]
                parsed_od = processor.post_process_generation(pred_od, task="<OD>",
                                                              image_size=(frame.shape[1], frame.shape[0]))

            # --- AFFICHAGE PROPRE DANS LE TERMINAL ---
            timestamp = time.strftime("%H:%M:%S")
            print("-" * 60)
            print(f"🕒 HEURE : {timestamp}")
            print(f"📝 CONTEXTE : {description}")

            if "labels" in parsed_od and parsed_od["labels"]:
                obj_list = ", ".join(set(parsed_od["labels"]))  # set() pour éviter les doublons
                print(f"📦 OBJETS DÉTECTÉS : {obj_list}")
            print("-" * 60)

        except Exception as e:
            print(f"⚠️ Erreur analyse : {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()