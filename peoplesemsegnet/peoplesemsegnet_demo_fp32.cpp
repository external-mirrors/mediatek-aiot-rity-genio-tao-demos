#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/core/ocl.hpp>
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"
#include "absl/cleanup/cleanup.h"
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/delegate_loader.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/tflite_settings_json_parser.h"
#include <cmath>

using namespace cv;
using namespace std;
using namespace tflite;

int model_width;
int model_height;
int model_channels;

std::unique_ptr<Interpreter> interpreter;

struct RGB {
    unsigned char blue;
    unsigned char green;
    unsigned char red;
};

const RGB Colors[2] = {
    {0, 50, 0},    // Background: Green
    {0, 100, 255}     // Person: Red
};

void GetImageTFLite(float* out, Mat &src) {
    if (out == nullptr) {
        throw std::runtime_error("Output tensor is null");
    }

    static Mat image;
    cv::resize(src, image, Size(model_width, model_height), INTER_LINEAR);
    image.convertTo(image, CV_32F, 1.0 / 127.5, -1.0);

    float *in = (float*)image.data;
    int len = image.rows * image.cols * image.channels();
    for (int i = 0; i < len; i++) {
        out[i] = in[i];
    }
}

void detect_from_video(Mat &src) {
    static Mat frame(model_height, model_width, CV_8UC3);
    static Mat blend(src.rows, src.cols, CV_8UC3);

    float* input_tensor = interpreter->typed_tensor<float>(interpreter->inputs()[0]);
    if (input_tensor == nullptr) {
        throw std::runtime_error("Input tensor is null");
    }

    GetImageTFLite(input_tensor, src);

    if (interpreter->Invoke() != kTfLiteOk) {
        throw std::runtime_error("Failed to invoke tflite interpreter");
    }

    int64_t* data = interpreter->typed_output_tensor<int64_t>(0);
    if (data == nullptr) {
        throw std::runtime_error("Output tensor is null");
    }

    RGB* rgb = (RGB*)frame.data;
    for (int i = 0; i < model_height; i++) {
        for (int j = 0; j < model_width; j++) {
            int mi = data[i * model_width + j];
            rgb[j + i * model_width] = Colors[mi];
        }
    }

    cv::resize(frame, blend, Size(src.cols, src.rows), INTER_LINEAR);
    cv::addWeighted(src, 0.5, blend, 0.5, 0.0, src);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <cam_id>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string cam_id = argv[2];

    std::string pipeline = "v4l2src device=/dev/video" + cam_id + " ! video/x-raw,width=1920,height=1080 ! v4l2convert ! video/x-raw,width=960,height=544,format=BGR ! appsink drop=true sync=false";
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video stream." << std::endl;
        return 1;
    }

    std::unique_ptr<FlatBufferModel> model = FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        throw std::runtime_error("Failed to load TFLite model");
    }

    using tflite::delegates::utils::LoadDelegateFromSharedLibrary;
    using tflite::delegates::utils::TfLiteSettingsJsonParser;
    constexpr char kSampleDelegatePath[] = "/usr/lib/libneuron_stable_delegate.so";
    constexpr char kSettingsPath[] = "/usr/share/label_image/stable_delegate_settings.json";

    const TfLiteStableDelegate* stable_delegate = LoadDelegateFromSharedLibrary(kSampleDelegatePath);
    if (stable_delegate == nullptr || stable_delegate->delegate_plugin == nullptr)
        throw std::runtime_error("Failed to load stable delegate from library");

    TfLiteSettingsJsonParser parser;
    const tflite::TFLiteSettings* settings = parser.Parse(kSettingsPath);
    if (settings == nullptr)
        throw std::runtime_error("Failed to load json settings");

    TfLiteOpaqueDelegate* opaque_delegate = stable_delegate->delegate_plugin->create(settings);
    if (stable_delegate == nullptr)
        throw std::runtime_error("Failed to create opaque delegate");

    ops::builtin::BuiltinOpResolver op_resolver;
    InterpreterBuilder(*model, op_resolver)(&interpreter);

    if (interpreter->ModifyGraphWithDelegate(opaque_delegate) != kTfLiteOk)
        throw std::runtime_error("Failed to modify graph with Opaque delegate");

    interpreter->AllocateTensors();
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4);

    int In = interpreter->inputs()[0];
    model_height = interpreter->tensor(In)->dims->data[1];
    model_width = interpreter->tensor(In)->dims->data[2];
    model_channels = interpreter->tensor(In)->dims->data[3];
    cout << "height   : " << model_height << endl;
    cout << "width    : " << model_width << endl;
    cout << "channels : " << model_channels << endl;

    cout << "Start grabbing, press Ctrl+C to terminate" << endl;
   
    // Create a window and set it to full screen
    namedWindow("Output", WINDOW_NORMAL);
    setWindowProperty("Output", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

    Mat frame;
    while (true) {
        bool ret = cap.read(frame);
        if (!ret) {
            cerr << "Error: Could not grab frame from camera." << endl;
            break;
        }

        detect_from_video(frame);

        // Resize the output to 1920x1080
        //cv::resize(frame, frame, Size(1920, 1080), INTER_LINEAR);

        // Show the frame
        imshow("Output", frame);

        char esc = waitKey(5);
        if (esc == 27) break;
    }

    cout << "Closing the camera" << endl;
    cap.release();
    destroyAllWindows();

    if (opaque_delegate != nullptr) {
        stable_delegate->delegate_plugin->destroy(opaque_delegate);
        opaque_delegate = nullptr;
    }

    interpreter.reset();

    cout << "Bye!" << endl;

    return 0;
}
