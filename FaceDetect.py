import os
import numpy as np
import cv2
import json
import pickle
from typing import Tuple
from deepface import DeepFace
import time

# Constants
FACE_DB = "faces"
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
RECOGNITION_THRESHOLD = 0.65
ENCODINGS_FILE = "face_encodings.pkl"
FRAME_INTERVAL = 0.1

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Flattened lists for encodings and names
# known_face_encodings[i] corresponds to known_face_names[i]
known_face_encodings = []
known_face_names = []

# For controlling the frequency of recognition calls
last_process_time = 0
last_recognition_result = ("Unknown", 1.0)

def load_known_faces() -> None:
    """
    Load known faces from the pickle file. If none is found, process the face DB.
    """
    global known_face_encodings, known_face_names

    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                data = pickle.load(f)
                known_face_encodings = data['encodings']  # List of numpy arrays
                known_face_names = data['names']          # List of labels
            print(f"Loaded {len(known_face_encodings)} existing face encodings.")
        except Exception as e:
            print(f"Error loading encodings: {e}")
            known_face_encodings = []
            known_face_names = []
    else:
        print("No existing encodings found. Processing faces...")
        face_processing()

def load(ls: list) -> None:
    """
    Create and save face labels dictionary for reference (optional).
    """
    dct = {ls.index(i): i for i in ls}
    with open("FaceLabels.json", 'w') as f:
        json.dump(dct, f, indent=4)

def getFaceLabel(ind: int) -> str:
    """
    Get face label from saved dictionary (optional utility).
    """
    try:
        with open("FaceLabels.json", "r") as f:
            dct = json.load(f)
        return dct[str(ind)]
    except:
        return 'Unknown'

def face_processing() -> None:
    """
    Process faces in FACE_DB and save encodings to a pickle file.
    Stores all encodings in a single list and the corresponding names in a parallel list.
    """
    global known_face_encodings, known_face_names

    try:
        people_list = [x for x in os.listdir(FACE_DB) if not x.startswith('.')]
        if not people_list:
            print("No people found in faces directory.")
            return

        print(f"Processing faces for {len(people_list)} people...")
        load(people_list)  # Save a label dictionary (optional)

        new_encodings = []
        new_names = []

        for person_folder in people_list:
            person_dir = os.path.join(FACE_DB, person_folder)
            if not os.path.isdir(person_dir):
                continue

            print(f"Processing {person_folder}")
            face_files = [f for f in os.listdir(person_dir) if not f.startswith('.')]

            for face_file in face_files:
                try:
                    image_path = os.path.join(person_dir, face_file)

                    # Obtain the embedding
                    embedding = DeepFace.represent(
                        img_path=image_path,
                        model_name="VGG-Face",
                        enforce_detection=True
                    )[0]["embedding"]
                    
                    # Normalize the embedding (important for cosine similarity)
                    embedding = embedding / np.linalg.norm(embedding)

                    # Append the embedding and corresponding name
                    new_encodings.append(embedding)
                    new_names.append(person_folder)
                    print(f"Added encoding for {person_folder}")

                except Exception as e:
                    print(f"Error processing {face_file} in {person_folder}: {e}")
                    continue

        # Update global lists
        known_face_encodings = new_encodings
        known_face_names = new_names

        # Save them to file
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump({
                'encodings': known_face_encodings,
                'names': known_face_names
            }, f)

        print(f"Saved encodings for {len(known_face_names)} faces.")

    except Exception as e:
        print(f"Face processing error: {e}")

def recog(face_image: np.ndarray, threshold: float) -> Tuple[str, float]:
    """
    Recognize a face and return (name, confidence).
    Uses vectorized cosine similarity across all known embeddings.
    Caches results for frames if processed within FRAME_INTERVAL seconds.
    """
    global last_process_time, last_recognition_result

    current_time = time.time()
    # Skip re-processing if last recognition was too recent
    if current_time - last_process_time < FRAME_INTERVAL:
        return last_recognition_result

    try:
        # Ensure encodings are loaded
        if not known_face_encodings:
            load_known_faces()
        if not known_face_encodings:
            return "Unknown", 1.0

        # Preprocess face for DeepFace
        face_image = cv2.resize(face_image, (160, 160))

        # Get embedding
        try:
            embedding = DeepFace.represent(
                img_path=face_image,
                model_name="VGG-Face",
                enforce_detection=False  # We already have the face region
            )[0]["embedding"]
            embedding = embedding / np.linalg.norm(embedding)
        except Exception as e:
            print(f"Embedding error: {e}")
            return "Unknown", 1.0

        # Convert known_face_encodings into a NumPy array (if not already)
        # Each row is one embedding
        encodings_matrix = np.vstack(known_face_encodings)  # shape: (N, D)
        # Dot product with your single embedding (D,) => returns shape (N,)
        similarities = encodings_matrix.dot(embedding)

        # Find best match
        max_index = np.argmax(similarities)
        max_similarity = similarities[max_index]
        best_match = known_face_names[max_index]

        last_process_time = current_time
        if max_similarity > threshold:
            last_recognition_result = (best_match, float(max_similarity))
            return best_match, float(max_similarity)

        last_recognition_result = ("Unknown", 1.0)
        return "Unknown", 1.0

    except Exception as e:
        print(f"Recognition error: {e}")
        return "Unknown", 1.0

def main():
    """
    Main loop: Opens camera, detects faces, and runs recognition.
    Press 'q' to exit.
    """
    global known_face_encodings, known_face_names

    print("Initializing face recognition system...")
    load_known_faces()
    # If encodings are still empty, try processing the face DB
    if not known_face_encodings:
        print("No known faces. Processing face database...")
        face_processing()
        load_known_faces()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received from camera. Exiting...")
            break

        # Flip frame horizontally for a mirror-like view
        frame = cv2.flip(frame, 1)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS
        )

        for (x, y, w, h) in faces:
            # Enlarge bounding box a bit (optional)
            x1 = max(0, x - int(w * 0.1))
            y1 = max(0, y - int(h * 0.1))
            x2 = min(frame.shape[1], x + w + int(w * 0.1))
            y2 = min(frame.shape[0], y + h + int(h * 0.1))

            # Extract face region
            face_image = frame[y1:y2, x1:x2]

            # Recognize face
            name, confidence = recog(face_image, RECOGNITION_THRESHOLD)

            # Draw bounding box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} ({confidence:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()