# GENIO TAO DEMOS

This repository consists of demo C++ example code using converted TAO models mentioned in documentation

### 1. People Detection

Consists of the TFLite model, label file and CPP code for NPU execution. Execute as below:

`./people_detection <model_name> <label_file_name> <cam_id>`

### 2. License Plate Detection

Consists of the TFLite model, label file and CPP code for NPU execution. Execute as below:

`./license_plate_detection <model_name> <label_file_name> <input_video.mp4> <output_video.mp4>`

### 3. BodyPoseNet

Consists of the TFLite model, and CPP code for NPU execution. Execute as below:

`./bodyposenet <model_name> <cam_id>`

### 4. Retail Object Recognition

Consists of the CSV logistic regression and CPP code for NPU execution. Logistic regression trained on 
a custom dataset of coca-cola and pepsi images. The application recognizes coca-cola and pepsi cans and outputs the prices, simulating a retail checkout. 
Execute as below:

`./retail_object_recognition <model_path> <csv_path> <output_video>`

### 5. Action Recogntion

Python Demo to execute action recognition on CPU. CPP code for NPU execution will be added later. 
Execute as:

`python3 action_recognition.py --model <model_name> --video <path_to_video_file>`

