import os
import numpy as np
import cv2
import json

# Constants
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
FACE_DB = "faces"
scalefactor = 1.1
minneighbors = 7
cropped_face_location = "cropped_faces.npy"
cropped_label_location = "cropped_labels.npy"
RECOGNITION_THRESHOLD = 90

# Initialize face detector and recognizer globally
face_detector = cv2.CascadeClassifier(cascPathface)
face_recognizer = None  # Will be initialized later

def train(cropped_face_location, cropped_label_location):
    """Train the face recognizer using saved face data"""
    global face_recognizer  # Use global variable
    
    try:
        if not os.path.exists(cropped_face_location) or not os.path.exists(cropped_label_location):
            print("Training files not found. Please add face images and run face_processing first.")
            return None
            
        train_faces = np.load(cropped_face_location, allow_pickle=True)
        train_labels = np.load(cropped_label_location, allow_pickle=True)
        
        if len(train_faces) == 0:
            print("No training faces found")
            return None
            
        # Convert labels to int32
        train_labels = np.array(train_labels, dtype=np.int32)
        
        # Create recognizer with adjusted parameters
        face_recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
        
        # Normalize training faces
        normalized_faces = []
        for face in train_faces:
            if face is not None and face.size > 0:
                face = cv2.resize(face, (100, 100))
                face = cv2.equalizeHist(face)
                normalized_faces.append(face)
        
        if normalized_faces:
            face_recognizer.train(np.array(normalized_faces), train_labels)
            print(f"Face recognizer trained successfully with {len(normalized_faces)} faces")
            return face_recognizer
        else:
            print("No valid faces for training")
            return None
            
    except Exception as e:
        print(f"Training error: {e}")
        return None

def load(ls):
    """Create and save face labels dictionary"""
    dct = {}
    for i in ls:
        dct[ls.index(i)] = i
    with open("FaceLabels.json", 'w') as f:
        json.dump(dct, f, indent=4)

def getFaceLabel(ind):
    """Get face label from saved dictionary"""
    try:
        with open("FaceLabels.json", "r") as f:
            dct = json.load(f)
        return dct[str(ind)]
    except:
        return 'Unknown'

def face_processing():
    """Process faces and save training data"""
    train_faces = []
    train_label = []
    
    try:
        LIST_OF_PEOPLE = [x for x in os.listdir(FACE_DB) if not x.startswith('.')]
        if not LIST_OF_PEOPLE:
            print("No people found in faces directory")
            return
            
        print(f"Processing faces for {len(LIST_OF_PEOPLE)} people")
        load(LIST_OF_PEOPLE)
        
        for index, person_folder in enumerate(LIST_OF_PEOPLE):
            person_dir = os.path.join(FACE_DB, person_folder)
            print(f"Processing {person_folder}")
            
            for face_file in os.listdir(person_dir):
                try:
                    image_path = os.path.join(FACE_DB, person_folder, face_file)
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                        
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)  # Enhance contrast
                    
                    faces = face_detector.detectMultiScale(
                        gray,
                        scaleFactor=scalefactor,
                        minNeighbors=minneighbors,
                        minSize=(30, 30)
                    )
                    
                    for face in faces:
                        left, top, width, height = face
                        face_roi = gray[top:top+height, left:left+width]
                        face_roi = cv2.resize(face_roi, (100, 100))
                        face_roi = cv2.equalizeHist(face_roi)
                        train_faces.append(face_roi)
                        train_label.append(index)
                        print(f"Added face for {person_folder}")
                        
                except Exception as e:
                    print(f"Error processing {face_file}: {e}")
                    continue
        
        if train_faces:
            print(f"Saving {len(train_faces)} faces for training")
            np.save(cropped_face_location, np.array(train_faces), allow_pickle=True)
            np.save(cropped_label_location, np.array(train_label), allow_pickle=True)
            print("Training data saved successfully")
        else:
            print("No faces found for training")
            
    except Exception as e:
        print(f"Face processing error: {e}")

def recog(cropped_face, threshold):
    """Recognize a face and return name"""
    global face_recognizer  # Use global variable
    
    try:
        if face_recognizer is None:
            print("Face recognizer not initialized. Training now...")
            face_recognizer = train(cropped_face_location, cropped_label_location)
            if face_recognizer is None:
                return "Unknown", threshold
        
        if cropped_face is None or cropped_face.size == 0:
            return "Unknown", threshold
            
        # Preprocess face
        cropped_face = cv2.resize(cropped_face, (100, 100))
        cropped_face = cv2.equalizeHist(cropped_face)
        
        # Get prediction
        label, confidence = face_recognizer.predict(cropped_face)
        print(f"Prediction: Label={label}, Confidence={confidence}")
        
        if confidence < threshold:
            name = getFaceLabel(label)
            print(f"Recognized as {name}")
            return name, confidence
        else:
            print(f"Unknown face (confidence too high: {confidence})")
            return "Unknown", confidence
            
    except Exception as e:
        print(f"Recognition error: {e}")
        if face_recognizer is None:
            print("Attempting to retrain face recognizer...")
            face_recognizer = train(cropped_face_location, cropped_label_location)
        return "Unknown", threshold

# Initialize face recognition system
print("Initializing face recognition system...")
face_processing()
face_recognizer = train(cropped_face_location, cropped_label_location)

if face_recognizer is None:
    print("Initial face recognizer training failed. Will attempt to train when needed.")
else:
    print("Face recognition system ready")

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=scalefactor,
            minNeighbors=minneighbors
        )
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            name, conf = recog(face_roi, RECOGNITION_THRESHOLD)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()