import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

# ÉTAPE 1 : On dit à Transformers d'utiliser le format sécurisé
# et on désactive la vérification stricte de la version de torch
os.environ["TRANSFORMERS_VERIFY_SCHEDULED_REMOVAL"] = "false"

# ÉTAPE 2 : Charger le modèle en précisant l'utilisation de safetensors
model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name, use_safetensors=True)

print("Modèle chargé avec succès !")
# Initialiser la caméra
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        # Convertir en image PIL
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Générer la description
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs)
        description = processor.decode(outputs[0], skip_special_tokens=True)
        print(description)

        # Afficher le frame
        cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()