# Importing the necessary libraries
import cv2
import os

# %%
#Create a directory to save the images if it doesn't already exist
if not os.path.exists("Selfie_Images"):
    os.mkdir("Selfie_Images")

# %%
# Initialize the webcam (0 means the default camera)
camera = cv2.VideoCapture(0)

# %%
# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# %%
 #Counter to keep track of the number of selfies taken
selfie_count = 0

while selfie_count < 5:
    # Capture a frame from the webcam
    ret, frame = camera.read()
    if not ret:
        break
    
    # Convert the frame to grayscale (required by the face detection model)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected faces and save each face
    for i, (x, y, w, h) in enumerate(faces):
        # Crop the face from the frame
        face = frame[y:y+h, x:x+w]
        # Create a separate folder for each face if multiple faces are detected
        face_folder = f"Selfie_Images/Face_{i+1}"
        if not os.path.exists(face_folder):
            os.mkdir(face_folder)
        # Save the cropped face image
        face_image_path = os.path.join(face_folder, f"selfie_{selfie_count+1}.jpg")
        cv2.imwrite(face_image_path, face)
    
    # Show the frame with rectangles around faces
    cv2.imshow("Webcam - Press 'q' to Quit", frame)
    
    # Wait for a key press or a delay to take the next selfie
    key = cv2.waitKey(1000)  # Wait 1 second before taking next selfie
    selfie_count += 1

    # If 'q' is pressed, exit early
    if key == ord('q'):
        break

# %%
# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()

# %%
# Edge Detection
edge_image = cv2.Canny(frame, 100, 200)

# Display original vs edge detected image
cv2.imshow("Original Image", frame)
cv2.imshow("Edge Detected Image", edge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Image Sharpening
import numpy as np
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_image = cv2.filter2D(frame, -1, kernel)

# Display original vs sharpened image
cv2.imshow("Original Image", frame)
cv2.imshow("Sharpened Image", sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Image Sharpening
import numpy as np
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_image = cv2.filter2D(frame, -1, kernel)

# Display original vs sharpened image
cv2.imshow("Original Image", frame)
cv2.imshow("Sharpened Image", sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Blur Image
blurred_image = cv2.GaussianBlur(frame, (5, 5), 0)

# Display original vs blurred image
cv2.imshow("Original Image", frame)
cv2.imshow("Blurred Image", blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
 #Image Resize
resized_image = cv2.resize(frame, None, fx=0.5, fy=0.5)

# Display original vs resized image
cv2.imshow("Original Image", frame)
cv2.imshow("Resized Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Image Rotation
(h, w) = frame.shape[:2]
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_image = cv2.warpAffine(frame, rotation_matrix, (w, h))

# Display original vs rotated image
cv2.imshow("Original Image", frame)
cv2.imshow("Rotated Image", rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Image Augmentation
flipped_image = cv2.flip(frame, 1)  # Horizontal flip
(h, w) = flipped_image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), -30, 1.0)
augmented_image = cv2.warpAffine(flipped_image, rotation_matrix, (w, h))
resized_augmented = cv2.resize(augmented_image, (200, 200))

# Display original vs augmented image
cv2.imshow("Original Image", frame)
cv2.imshow("Augmented Image", resized_augmented)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Image Cropping
cropped_image = frame[50:200, 100:300]

# Display original vs cropped image
cv2.imshow("Original Image", frame)
cv2.imshow("Cropped Image", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Convert Image to Black & White
bw_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Create a Negative of the Image
negative_image = 255 - frame

# Display original vs black & white vs negative image
cv2.imshow("Original Image", frame)
cv2.imshow("Black & White Image", bw_image)
cv2.imshow("Negative Image", negative_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Face Detection
faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display original vs face detected image
cv2.imshow("Original Image", gray_frame)
cv2.imshow("Face Detected Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Identifying Facial Features
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
eyes = eye_cascade.detectMultiScale(gray_frame, 1.1, 10)
for (ex, ey, ew, eh) in eyes:
    cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

# Display original vs eyes detected image
cv2.imshow("Original Image", gray_frame)
cv2.imshow("Eyes Detected Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
