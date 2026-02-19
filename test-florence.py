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

# Forçage sur CPU pour Mac Intel i9
device = "cpu"
model_id = "microsoft/Florence-2-base"

print(f"--- INITIALISATION SUR MAC INTEL i9 ---")

try:
    print(f"Étape 1: Chargement de la configuration...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Étape cruciale pour éviter l'erreur de cache sur CPU
    config.use_cache = False

    print(f"Étape 2: Chargement du processeur et du modèle...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # On force l'implémentation 'eager' pour la compatibilité Intel
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        attn_implementation="eager"
    ).to(device)

    print("✅ Florence-2 est prêt !")

except Exception as e:
    print(f"❌ Erreur critique au chargement : {e}")
    exit()

# --- BOUCLE DE CAPTURE ---
cap = cv2.VideoCapture(0)
capture_interval = 8  # On laisse 8 secondes au i9 pour chaque analyse
last_capture_time = time.time()

print("\n📷 Caméra active. Appuyez sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Affichage du retour vidéo
    cv2.imshow('Vision IA - Appuyez sur Q pour quitter', frame)

    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        last_capture_time = current_time

        try:
            # 1. Préparation de l'image (optimisée pour CPU)
            # On réduit la taille pour soulager le processeur
            frame_resized = cv2.resize(frame, (480, 270))
            image_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

            # 2. Préparation des entrées
            # Utiliser <CAPTION> au lieu de <DETAILED_CAPTION> si c'est trop lent
            prompt = "<CAPTION>"
            inputs = processor(text=prompt, images=image_pil, return_tensors="pt").to(device)

            # 3. Génération SÉCURISÉE (Correctif du crash AttributeError)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=100,
                    num_beams=1,
                    use_cache=False,  # <--- Empêche le crash 'NoneType'
                    do_sample=False  # <--- Stabilise la sortie sur CPU
                )

            # 4. Décodage
            prediction = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] IA dit : {prediction}")

        except Exception as e:
            print(f"⚠️ Erreur lors de l'analyse : {e}")

    # Sortie du programme
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- NETTOYAGE ---
cap.release()
cv2.destroyAllWindows()
print("Programme arrêté proprement.")