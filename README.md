# Face Mask Detection in terminal area of manufacturing plant
## Architecture
![FaceMaskDetection workflow](architecture.PNG)

## Method
Classification model:
- Yolov5 classify backbone(cspdarknet53)

Face detection model:
- Face detection: Haarfeature algorithm(Opencv)

ONNX runtime:
- Convert ONNX format for low latency

More information:
https://docs.ultralytics.com/models/yolov5/

## Achievement
- Running realtime on CPU-base laptop
- Ready for CPU edge device
- Good classification mask or no mask
## Limitation
- Face detection is not good at side view
- Improve with strong model about face detection to handle this problem
