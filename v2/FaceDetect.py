import os
import numpy as np
import cv2
import json

cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
FACE_DB = "faces"
scalefactor = 1.1
minneighbors = 7
cropped_face_location = "cropped_faces.npy"
cropped_label_location = "cropped_labels.npy"

# Lower threshold for better recognition
RECOGNITION_THRESHOLD = 80  # Try a lower value if recognition is too strict

def train(cropped_face_location, cropped_label_location):
    train_faces = np.load(cropped_face_location, allow_pickle=True)
    train_labels = np.load(cropped_label_location, allow_pickle=True)
    face_recogniser = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
        threshold=RECOGNITION_THRESHOLD
    )
    face_recogniser.train(train_faces, np.array(train_labels))
    return face_recogniser

def load(ls):
    dct = {}
    for i in ls:
        dct[ls.index(i)]=i
    dct = json.dumps(dct,indent=4)
    with open("FaceLabels.json",'w') as f:
        f.write(dct)
def getFaceLabel(ind):
    try:
        with open("FaceLabels.json","r") as f:
            dct = json.load(f)
        return dct[str(ind)]
    except:
        return 'Unknown'

face_detector=cv2.CascadeClassifier(cascPathface)
def face_processing():
    train_faces = []
    train_label = []
    LIST_OF_PEOPLE = [x for x in os.listdir(FACE_DB) if x!=".DS_Store"]
    load(LIST_OF_PEOPLE)
    
    for index, person_folder in enumerate(LIST_OF_PEOPLE):
        for face_file in os.listdir(os.path.join(FACE_DB, person_folder)): 
            try:
                image = cv2.imread(os.path.join(FACE_DB, person_folder, face_file))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Apply histogram equalization
                gray = cv2.equalizeHist(gray)
                faces = face_detector.detectMultiScale(gray, scaleFactor=scalefactor, minNeighbors=minneighbors)
                
                for face in faces:
                    left, top, width, height = face
                    cropped_face = gray[top:top+height, left:left+width]
                    # Resize to consistent size
                    cropped_face = cv2.resize(cropped_face, (200, 200))
                    train_faces.append(cropped_face)
                    train_label.append(index)
            except Exception as e:
                print(f"Error occurred: {e}")
                
    if train_faces:
        np.save("cropped_faces.npy", np.array(train_faces))
        np.save("cropped_labels.npy", np.array(train_label))
    else:
        print("No faces detected in training data")

def recog(cropped_face, threshold):
    try:
        # Resize input face to match training size
        cropped_face = cv2.resize(cropped_face, (200, 200))
        # Apply histogram equalization
        cropped_face = cv2.equalizeHist(cropped_face)
        
        # Debug prints
        print("Attempting recognition...")
        label, confidence = face_recognizer.predict(cropped_face)
        print(f"Recognition result - Label: {label}, Confidence: {confidence}")
        
        if confidence < threshold:
            name = getFaceLabel(label)
            print(f"Recognized as: {name}")
            return name, confidence
        print("Confidence too low - marked as Unknown")
        return "Unknown", threshold
    except Exception as e:
        print(f"Recognition error: {e}")
        return "Unknown", threshold

def track_faces():       
    k = 65
    cam = cv2.VideoCapture(0)
    
    while(k not in (27, ord('q'), ord('Q'))):
        ret, frames = cam.read()
        frames = cv2.flip(frames, 1)
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = face_detector.detectMultiScale(gray, scaleFactor=scalefactor, minNeighbors=minneighbors)
        
        if len(faces):
            for (left, top, width, height) in faces:
                face_roi = gray[top:top+height, left:left+width]
                detected_name, confidence = recog(face_roi, RECOGNITION_THRESHOLD)
                
                cv2.rectangle(frames, (left, top), (left+width, top+height), (10, 0, 255), 2)
                frames = cv2.putText(frames, detected_name, (left, top-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 0, 255), 2)
        
        cv2.imshow('Face Recognition', frames)
        k = cv2.waitKey(1)
        
    cam.release()
    cv2.destroyAllWindows()

# Initialize face detection and recognition
face_detector = cv2.CascadeClassifier(cascPathface)
face_processing()
face_recognizer = train(cropped_face_location, cropped_label_location)

if __name__ == "__main__":
    track_faces()