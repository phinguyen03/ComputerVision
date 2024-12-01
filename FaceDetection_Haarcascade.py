import cv2
import torch
from torchvision import transforms
from PIL import Image


# Load a model
model = torch.hub.load('F:\document\AI\FaceDetection\yolov5', 'custom', path='best.onnx', source='local')

# Preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((320, 320)),  # YOLOv5 classification expects 320x320 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(0.5)
])


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using a face detection method (Haar feature)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Extract the face ROI
        face = frame[y:y+h, x:x+w] # bouding box
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)) # face detection 
        
        # Preprocess image
        input_tensor = transform(face_pil).unsqueeze(0) # 4Dimensions

        # Inference
        results = model(input_tensor)
        predicted = torch.argmax(results, dim=1)
        label = "Mask" if predicted.item() == 0 else "No Mask"

        # Draw bounding box and label
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
