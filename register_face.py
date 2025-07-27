import cv2
import os

# Path to store images
dataset_path = 'dataset'

# Create the folder if it doesn't exist
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Get user name/id to label the image
user_name = input("Enter name or ID for registration: ").strip()
user_folder = os.path.join(dataset_path, user_name)

if not os.path.exists(user_folder):
    os.makedirs(user_folder)

# Open the webcam
cap = cv2.VideoCapture(0)
count = 0

print("📸 Capturing face images. Look at the camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    # Detect face using Haar cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw and save face
    for (x, y, w, h) in faces:
        count += 1
        face = frame[y:y+h, x:x+w]
        cv2.imwrite(f"{user_folder}/{count}.jpg", face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Register Face', frame)

    if cv2.waitKey(1) == 27 or count >= 20:  # ESC or 20 images
        break

print(f"✅ Registration completed for {user_name} with {count} images.")
cap.release()
cv2.destroyAllWindows()
