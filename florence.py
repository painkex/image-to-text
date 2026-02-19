import cv2
import time
from neo4j import GraphDatabase
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

# Connexion à Neo4j
uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"
driver = GraphDatabase.driver(uri, auth=(user, password))

# Charger Florence-2
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(device)

# Initialiser la webcam
cap = cv2.VideoCapture(0)  # 0 pour la webcam par défaut, ajustez si nécessaire
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

capture_interval = 5  # Analyse toutes les 5 secondes
last_capture_time = time.time()

# Fonction pour stocker une description dans Neo4j
def create_description(tx, timestamp, description):
    tx.run("CREATE (d:Description {timestamp: $timestamp, description: $description})",
           timestamp=timestamp, description=description)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la capture.")
        break

    # Afficher le flux en direct
    cv2.imshow('Flux en direct', frame)

    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        last_capture_time = current_time

        # Convertir le frame en image PIL pour Florence-2
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(text="<DETAILED_CAPTION>", images=image, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=1024, num_beams=3)
        description = processor.decode(outputs[0], skip_special_tokens=True)

        # Horodatage
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # Stocker dans Neo4j
        with driver.session() as session:
            session.write_transaction(create_description, timestamp, description)

        print(f"[{timestamp}] {description}")

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
driver.close()