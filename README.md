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

**Note: The demo works well for single person in the frame. The update for more than 1 people in frame to follow soon**

### 4. Retail Object Recognition

Consists of the CSV logistic regression and CPP code for NPU execution. Logistic regression trained on 
a custom dataset of coca-cola and pepsi images. The application recognizes coca-cola and pepsi cans and outputs the prices, simulating a retail checkout. 
Execute as below:

`./retail_object_recognition <model_path> <csv_path> <output_video>`

### 5. Action Recogntion

Python Demo to execute action recognition on CPU. CPP code for NPU execution will be added later. 
Execute as:

`python3 action_recognition.py --model <model_name> --video <path_to_video_file>`

### 6. People Semantic Segmentation

Consists of the TFLite modle and CPP doe for NPU execution. Execute as:

`./peoplesemsegnet_demo <model_file> <input_video> <output_video>`

## Compilation on Genio Target

The demos can be compiled on the Genio Target itself with necessary libraries. Add `IMAGE_INSTALL:append = "abseil-cpp"` to local.conf at time of image build and ensure `g++` is installed.
Follow the below command:

`g++ <script.cpp> -o <executable_name> -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_dnn -lopencv_flann -ltensorflowlite -ltensorflowlite_c -ldl -lflatbuffers -pthread -ldelegate_loader -ltflite_settings_json_parser -labsl_base -I/usr/include/opencv4/`

You can add/remove the opencv shared libraries based on functions used in the cpp code. 

Do Note: The code is specific to the model. If converting a new model with different input/output dtype, please edit the demo code accordingly.

## Genio TAO Documentation

Genio TAO Documentation can be accessed [here](https://mediatek.gitlab.io/genio/doc/tao/index.html) 