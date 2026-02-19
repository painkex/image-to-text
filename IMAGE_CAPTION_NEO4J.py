import cv2
import time
from neo4j import GraphDatabase
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Connexion à Neo4j
uri = "bolt://localhost:7687"  # Remplacez par votre URL si nécessaire
user = "neo4j"
password = "password"  # Remplacez par votre mot de passe
driver = GraphDatabase.driver(uri, auth=(user, password))

# Charger BLIP-2
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialiser la caméra
cap = cv2.VideoCapture(1)
capture_interval = 5  # Capture toutes les 5 secondes
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

    cv2.imshow('Flux en direct', frame)

    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        last_capture_time = current_time

        # Générer une description avec BLIP-2
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs)
        description = processor.decode(outputs[0], skip_special_tokens=True)

        # Horodatage
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # Stocker dans Neo4j
        with driver.session() as session:
            session.write_transaction(create_description, timestamp, description)

        print(f"[{timestamp}] {description}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
driver.close()