# Script pour le traitement d'images - Version 2.5
# 
# À vérifier avant de lancer le script :
# 1. Assurez-vous que les dossiers "images" (input_folder et output_folder) existent et contiennent les fichiers nécessaires.
# 2. Vérifiez que l'API LLaVA est en cours d'exécution sur http://localhost:11434.
# 3. Activez votre environnement virtuel avec : source myenv/bin/activate
# 4. Assurez-vous que les dépendances suivantes sont installées :
#    - rawpy
#    - PIL (Pillow)
#    - requests
#    - exiftool (et qu'il est accessible via /opt/homebrew/bin/exiftool).
# 5. Vérifiez que les fichiers RAW (.cr3) sont pris en charge par votre installation de rawpy.
# 6. Assurez-vous que le fichier de log ("script.log") peut être créé dans le répertoire courant.
# 7. Confirmez que les permissions d'écriture sont disponibles pour le dossier de sortie.
# 8. Testez une image manuellement pour valider le flux de traitement complet avant de lancer sur un grand lot.

import os
import requests
import json
import base64
import logging
from PIL import Image
import rawpy
import exiftool
import time

# Configuration
input_folder = "images"  # Remplacez par le chemin de votre dossier de photos
output_folder = "images/temp"  # Dossier où les fichiers texte seront sauvegardés
log_file = "script.log"

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Assurez-vous que le dossier de sortie existe
os.makedirs(output_folder, exist_ok=True)

def log_info(message):
    logging.info(message)

def log_error(message):
    logging.error(message)

# Fonction pour convertir un fichier RAW en JPEG de haute qualité
def convert_raw_to_jpeg(raw_path, jpeg_path, min_size=2000):
    try:
        with rawpy.imread(raw_path) as raw:
            rgb = raw.postprocess()
        
        image = Image.fromarray(rgb)
        width, height = image.size
        if width < min_size or height < min_size:
            ratio = min_size / min(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        image.save(jpeg_path, "JPEG", quality=90)
        log_info(f"Fichier RAW converti en JPEG : {jpeg_path}")
        return True
    except Exception as e:
        log_error(f"Erreur lors de la conversion du fichier RAW ({raw_path}) : {e}")
        return False

# Fonction pour encoder une image en base64
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        log_info(f"Encodage en base64 réussi pour : {image_path}")
        return image_base64
    except Exception as e:
        log_error(f"Erreur lors de l'encodage en base64 ({image_path}) : {e}")
        return None

# Fonction pour interroger LLaVA via Ollama
def query_llava(image_base64, prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llava:7B",
        "prompt": prompt,
        "images": [image_base64]
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers, stream=True)
        full_response = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))
                full_response += chunk.get("response", "")
                if chunk.get("done", False):
                    break
        log_info(f"Réponse de LLaVA reçue avec succès.")
        return full_response
    except Exception as e:
        log_error(f"Erreur lors de l'interrogation de LLaVA : {e}")
        return ""

# Fonction pour nettoyer et normaliser les tags
def clean_tags(tags_response):
    # Suppression des champs inutiles et normalisation des tags
    tags = [tag.strip() for tag in tags_response.split(",") if tag.strip()]
    clean_tags = []

    corrections = {
        "hair_color": "hair",
        "eye_color": "eyes",
        "scene": "",
        "lighting": "",
        "background": ""
    }

    for tag in tags:
        if ":" in tag:
            key, value = tag.split(":", 1)
            tag = value.strip()
        for old, new in corrections.items():
            tag = tag.replace(f"{old}_", new).strip()
        tag = tag.replace("_", " ").strip()
        clean_tags.append(tag)

    # Suppression des doublons
    return list(set(clean_tags))

# Fonction pour ajouter les tags aux métadonnées de l'image
def add_tags_to_image(image_path, tags):
    try:
        tags_str = ", ".join(tags)
        with exiftool.ExifTool(executable="/opt/homebrew/bin/exiftool") as et:
            et.execute(b"-XMP:Subject+=" + tags_str.encode("utf-8"), image_path.encode("utf-8"))
        log_info(f"Tags ajoutés aux métadonnées de l'image : {image_path}")
    except Exception as e:
        log_error(f"Erreur lors de l'ajout des tags aux métadonnées ({image_path}) : {e}")

# Fonction pour traiter une seule image
def process_image(image_path):
    try:
        filename = os.path.basename(image_path)
        base_filename, _ = os.path.splitext(filename)
        unique_id = str(int(time.time()))  # ID unique pour éviter les collisions
        log_info(f"Début du traitement de l'image : {filename}")

        # Conversion en preview si c'est un fichier RAW
        if filename.lower().endswith('.cr3'):
            jpeg_path = os.path.join(output_folder, f"{base_filename}_{unique_id}_preview.jpg")
            if not convert_raw_to_jpeg(image_path, jpeg_path):
                log_error(f"Échec de la conversion pour : {filename}")
                return
        else:
            jpeg_path = os.path.join(output_folder, f"{base_filename}_{unique_id}.jpg")
            Image.open(image_path).save(jpeg_path, "JPEG", quality=90)

        # Encodage en base64
        image_base64 = encode_image(jpeg_path)
        if not image_base64:
            log_error(f"Impossible d'encoder l'image : {filename}")
            return

        # Génération des tags
        prompt = (
            "Extract detailed attributes of the subject in this image, including: "
            "hair color, hair length, eye color, type of scene, and setting. "
            "Return the details as a comma-separated list of tags."
        )
        tags_response = query_llava(image_base64, prompt)
        #tags = clean_tags(tags_response)
        tags = tags_response
        # Sauvegarde des tags dans un fichier texte
        output_file = os.path.join(output_folder, f"{base_filename}_{unique_id}.txt")
        with open(output_file, "w") as f:
            f.write("\n".join(tags))

        # Ajout des tags aux métadonnées de l'image
        add_tags_to_image(jpeg_path, tags)

        log_info(f"Tags générés et enregistrés pour {filename} : {tags}")
        log_info(f"Fichiers générés : {jpeg_path}, {output_file}")
    except Exception as e:
        log_error(f"Erreur inattendue lors du traitement de l'image ({image_path}) : {e}")

# Fonction principale
if __name__ == "__main__":
    log_info("Démarrage du traitement des images.")

    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]

    for image_file in image_files:
        process_image(image_file)

    log_info("Traitement terminé.")
