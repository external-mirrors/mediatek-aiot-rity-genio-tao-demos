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
    {0, 0, 0},    // Background: Black
    {0, 0, 255}   // Person: Red
};

void GetImageTFLite(int8_t* out, Mat &src) {
    if (out == nullptr) {
        throw std::runtime_error("Output tensor is null");
    }

    int i;
    uint8_t *in;
    static Mat image;
    int Len;

    cv::resize(src, image, Size(model_width, model_height), INTER_NEAREST);

    in = image.data;
    Len = image.rows * image.cols * image.channels();
    for (i = 0; i < Len; i++) {
        out[i] = static_cast<int8_t>(in[i] - 128);
    }
}

void detect_from_video(Mat &src) {
    int i, j, mi;
    int64_t *data;
    RGB *rgb;
    static Mat image;
    static Mat frame(model_height, model_width, CV_8UC3);
    static Mat blend(src.rows, src.cols, CV_8UC3);

    int8_t* input_tensor = interpreter->typed_tensor<int8_t>(interpreter->inputs()[0]);
    if (input_tensor == nullptr) {
        throw std::runtime_error("Input tensor is null");
    }

    GetImageTFLite(input_tensor, src);

    if (interpreter->Invoke() != kTfLiteOk) {
        throw std::runtime_error("Failed to invoke tflite interpreter");
    }

    data = interpreter->typed_output_tensor<int64_t>(0);
    if (data == nullptr) {
        throw std::runtime_error("Output tensor is null");
    }

    rgb = (RGB *)frame.data;

    for (i = 0; i < model_height; i++) {
        for (j = 0; j < model_width; j++) {
            mi = data[i * model_width + j];
            rgb[j + i * model_width] = Colors[mi];
        }
    }

    cv::resize(frame, blend, Size(src.cols, src.rows), INTER_NEAREST);
    cv::addWeighted(src, 0.5, blend, 0.5, 0.0, src);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <input_video> <output_video>" << std::endl;
        return 1;
    }

    std::string model_file = argv[1];
    std::string input_video = argv[2];
    std::string output_video = argv[3];


    std::unique_ptr<FlatBufferModel> model = FlatBufferModel::BuildFromFile(model_file.c_str());

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

    std::string input_pipeline = "filesrc location=" + input_video + " ! parsebin ! v4l2h264dec ! v4l2convert ! video/x-raw,width=1920,height=1080,format=BGR ! appsink";

    cv::VideoCapture cap(input_pipeline, cv::CAP_GSTREAMER); // Open the specified video file using GStreamer pipeline
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return 1;
    }



    // Define the codec and create VideoWriter object using GStreamer pipeline
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    //std::string output_video = "output.mp4";
    std::string output_pipeline = "appsrc ! videoconvert ! queue ! v4l2h265enc extra-controls=\"controls,video_gop_size=10,video_bitrate=3600000,sequence_header_mode=1\" ! queue ! video/x-h265,level=(string)4.1,profile=main ! queue ! h265parse ! queue ! mp4mux ! queue ! filesink location=" + output_video;

    VideoWriter video_writer(output_pipeline, cv::CAP_GSTREAMER, 0, fps, Size(frame_width, frame_height), true);

    // Check if the VideoWriter object is initialized successfully
    if (!video_writer.isOpened()) {
        cerr << "Error: Could not open the output video file for write." << endl;
        return 1;
    }

    cout << "Start grabbing, press Ctrl+C to terminate" << endl;

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "End of movie" << endl;
            break;
        }

        detect_from_video(frame);

        // Write the frame into the file
        video_writer.write(frame);

        imshow("Output", frame);

        char esc = waitKey(5);
        if (esc == 27) break;
    }

    cout << "Closing the camera" << endl;
    cap.release();
    video_writer.release();
    destroyAllWindows();

    // Explicitly destroy the opaque delegate before exiting
    if (opaque_delegate != nullptr) {
        stable_delegate->delegate_plugin->destroy(opaque_delegate);
        opaque_delegate = nullptr;
    }

    // Explicitly reset the interpreter to ensure proper cleanup
    interpreter.reset();

    cout << "Bye!" << endl;

    return 0;
}

