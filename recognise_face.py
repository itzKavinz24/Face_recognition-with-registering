import os
import numpy as np
import cv2

# ========== Configuration ==========
dataset_path = 'dataset'
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
(width, height) = (130, 100)

# ========== Load Haar Cascade ==========
face_cascade = cv2.CascadeClassifier(haar_file)
if face_cascade.empty():
    raise IOError("Haar cascade file not found or failed to load!")

# ========== Training ==========
print("Training...")

(images, labels, names, id) = ([], [], {}, 0)

# Walk through dataset and load images
for (subdirs, dirs, files) in os.walk(dataset_path):
    for subdir in dirs:
        names[id] = subdir
        subject_path = os.path.join(dataset_path, subdir)
        for filename in os.listdir(subject_path):
            path = os.path.join(subject_path, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (width, height))
                images.append(img)
                labels.append(id)
        id += 1

# Convert to numpy arrays
if len(set(labels)) < 2:
    raise ValueError("Need at least two distinct classes for FisherFaceRecognizer to train!")

(images, labels) = [np.array(l) for l in (images, labels)]

# Create and train model
model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)
print("Training complete.")

# ========== Start Webcam for Recognition ==========
print("Starting face recognition...")
webcam = cv2.VideoCapture(0)
cnt = 0

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        label, confidence = model.predict(face_resize)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if confidence < 800:
            name = names[label]
            cv2.putText(frame, f'{name} - {confidence:.0f}', (x, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
            print("Recognized:", name)
            cnt = 0
        else:
            cnt += 1
            cv2.putText(frame, 'Unknown', (x, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
            if cnt > 100:
                print("Unknown person detected.")
                cv2.imwrite("unknown.jpg", frame)
                cnt = 0

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(10) == 27:  # ESC key to break
        break

webcam.release()
cv2.destroyAllWindows()
