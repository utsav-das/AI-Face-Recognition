import os
import imghdr
import numpy as np
from deepface import DeepFace
import tempfile
import logging

# ---------------------- Logging ----------------------
temp_dir = tempfile.gettempdir()
log_dir = os.path.join(temp_dir, "data", "log")
os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir, "generate_embedding_cache.log")
logging.basicConfig(filename=log_path, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# ---------------------- Configuration ----------------------
VERIFICATION_DIR = os.path.join(temp_dir, "data", "verification_image")
OUTPUT_FILE = os.path.join(temp_dir, "data", "threshold", "embedding_cache.npy")

# ---------------------- Main Script ----------------------
def generate_embedding_cache():
    embedding_dict = {}

    for person in os.listdir(VERIFICATION_DIR):
        person_path = os.path.join(VERIFICATION_DIR, person)
        if not os.path.isdir(person_path):
            continue

        embeddings = []
        for file in os.listdir(person_path):
            img_path = os.path.join(person_path, file)
            if not imghdr.what(img_path):
                continue

            try:
                result = DeepFace.represent(
                    img_path=img_path, 
                    model_name='ArcFace',
                    enforce_detection=False
                )[0]["embedding"]

                emb = np.array(result)
                emb = emb / np.linalg.norm(emb)
                embeddings.append(emb)
                logger.info(f"Processed: {img_path}")

            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")

        if embeddings:
            embedding_dict[person] = np.array(embeddings)

    np.save(OUTPUT_FILE, embedding_dict)
    logger.info(f"Saved embedding cache to {OUTPUT_FILE}")


if __name__ == "__main__":
    logger.info("Starting embedding cache generation...")
    generate_embedding_cache()
    logger.info("Embedding cache generation complete.")
