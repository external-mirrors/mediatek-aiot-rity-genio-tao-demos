#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <memory>
#include <iomanip>
#include <sstream>

#include "absl/cleanup/cleanup.h"
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/delegate_loader.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/tflite_settings_json_parser.h"
#include "tensorflow/lite/interpreter_builder.h"

// Function to preprocess the image
cv::Mat preprocess_image(const cv::Mat& image, const cv::Size& target_size, const TfLiteTensor* input_tensor) {
    cv::Mat resized_image;
    cv::resize(image, resized_image, target_size);
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);  // Normalize to [0, 1]

    // Quantize the image to int8
    float input_scale = input_tensor->params.scale;
    int input_zero_point = input_tensor->params.zero_point;
    resized_image = resized_image / input_scale + input_zero_point;
    cv::Mat quantized_image;
    resized_image.convertTo(quantized_image, CV_8S);  // Convert to int8

    return quantized_image;
}

// Extract keypoints from heatmap
std::pair<std::vector<cv::Point>, std::vector<float>> extract_keypoints(const cv::Mat& heatmap, float threshold = 0.1) {
    std::vector<cv::Point> keypoints(18, cv::Point(-1, -1));
    std::vector<float> probabilities(18, 0.0f);

    for (int i = 0; i < 18; ++i) {
        cv::Mat heatmap_slice;
        cv::extractChannel(heatmap, heatmap_slice, i);
        double min_val, max_val;
        cv::Point max_loc;
        cv::minMaxLoc(heatmap_slice, &min_val, &max_val, nullptr, &max_loc);
        probabilities[i] = static_cast<float>(max_val);
        if (max_val > threshold) {
            keypoints[i] = cv::Point(max_loc.x * 8, max_loc.y * 8);  // Scale to original image size
        }
    }
    return {keypoints, probabilities};
}

// Visualize the results
void visualize_pose(cv::Mat& image, const std::vector<cv::Point>& keypoints, const std::vector<std::pair<int, int>>& skeleton, float scale_x, float scale_y) {
    for (const auto& point : keypoints) {
        if (point.x != -1 && point.y != -1) {
            cv::circle(image, cv::Point(static_cast<int>(point.x * scale_x), static_cast<int>(point.y * scale_y)), 5, cv::Scalar(0, 255, 0), -1);
        }
    }
    for (const auto& edge : skeleton) {
        if (keypoints[edge.first - 1].x != -1 && keypoints[edge.second - 1].x != -1) {
            cv::line(image, 
                     cv::Point(static_cast<int>(keypoints[edge.first - 1].x * scale_x), static_cast<int>(keypoints[edge.first - 1].y * scale_y)), 
                     cv::Point(static_cast<int>(keypoints[edge.second - 1].x * scale_x), static_cast<int>(keypoints[edge.second - 1].y * scale_y)), 
                     cv::Scalar(255, 0, 0), 2);
        }
    }
}

void print_tensor_shape(const TfLiteTensor* tensor, const std::string& tensor_name) {
    std::cout << tensor_name << " shape: [";
    for (int i = 0; i < tensor->dims->size; ++i) {
        std::cout << tensor->dims->data[i];
        if (i < tensor->dims->size - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <input_video> <output_video>" << std::endl;
        return 1;
    }

    std::string model_file = argv[1];
    std::string input_video = argv[2];
    std::string output_video = argv[3];

    std::string input_pipeline = "filesrc location=" + input_video + " ! parsebin ! v4l2h264dec ! v4l2convert ! video/x-raw,width=1280,height=720,format=BGR ! appsink";

    cv::VideoCapture cap(input_pipeline, cv::CAP_GSTREAMER); // Open the specified video file using GStreamer pipeline
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return 1;
    }

    // Get video properties
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    // GStreamer pipeline for writing the output video
    std::string output_pipeline = "appsrc ! videoconvert ! queue ! v4l2h265enc extra-controls=\"controls,video_gop_size=10,video_bitrate=3600000,sequence_header_mode=1\" ! queue ! video/x-h265,level=(string)4,profile=main ! queue ! h265parse ! queue ! mp4mux ! queue ! filesink location=" + output_video; 

    // Define the codec and create VideoWriter object using GStreamer pipeline
    cv::VideoWriter video_writer(output_pipeline, cv::CAP_GSTREAMER, 0, fps, cv::Size(frame_width, frame_height), true);

    if (!video_writer.isOpened()) {
        std::cerr << "Error: Could not open the output video file for write." << std::endl;
        return 1;
    }

    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
    if (!model) {
        throw std::runtime_error("Failed to load TFLite model");
    }

    using tflite::delegates::utils::LoadDelegateFromSharedLibrary;
    using tflite::delegates::utils::TfLiteSettingsJsonParser;
    constexpr char kSampleDelegatePath[] = "/usr/lib/libneuron_stable_delegate.so";
    constexpr char kSettingsPath[] = "/usr/share/label_image/stable_delegate_settings.json";

    // Load stable delegate.
    const TfLiteStableDelegate* stable_delegate = LoadDelegateFromSharedLibrary(kSampleDelegatePath);
    if (stable_delegate == nullptr || stable_delegate->delegate_plugin == nullptr)
        throw std::runtime_error("Failed to load stable delegate from library");

    // Load settings.
    TfLiteSettingsJsonParser parser;
    const tflite::TFLiteSettings* settings = parser.Parse(kSettingsPath);
    if (settings == nullptr)
        throw std::runtime_error("Failed to load json settings");

    // Create opaque delegate.
    TfLiteOpaqueDelegate* opaque_delegate = stable_delegate->delegate_plugin->create(settings);
    if (stable_delegate == nullptr)
        throw std::runtime_error("Failed to create opaque delegate");
    absl::Cleanup destory_opaque_delegate = [&] {
        stable_delegate->delegate_plugin->destroy(opaque_delegate);
    };

    tflite::ops::builtin::BuiltinOpResolver op_resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, op_resolver)(&interpreter);

    if (interpreter->ModifyGraphWithDelegate(opaque_delegate) != kTfLiteOk)
        throw std::runtime_error("Failed to modify graph with Opaque delegate");

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Failed to allocate tensors");
    }

    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4);

    // Get input and output details
    TfLiteTensor* input_tensor = interpreter->tensor(interpreter->inputs()[0]);
    TfLiteTensor* output_affinity_tensor = interpreter->tensor(interpreter->outputs()[0]);
    TfLiteTensor* output_heatmap_tensor = interpreter->tensor(interpreter->outputs()[1]);

    // Print input and output shapes
    print_tensor_shape(input_tensor, "Input tensor");
    print_tensor_shape(output_affinity_tensor, "Output affinity tensor");
    print_tensor_shape(output_heatmap_tensor, "Output heatmap tensor");

    // Visualization parameters
    std::vector<std::string> keypoints_labels = {
        "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
        "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
        "left_eye", "right_ear", "left_ear"
    };
    std::vector<std::pair<int, int>> skeleton = {
        {2, 9}, {9, 10}, {10, 11}, {2, 12}, {12, 13}, {13, 14}, {2, 3}, {3, 4},
        {4, 5}, {3, 17}, {2, 6}, {6, 7}, {7, 8}, {6, 18}, {2, 1}, {1, 15}, {1, 16}, {15, 17}, {16, 18}
    };

    // Set the desired input size for the model
    cv::Size input_size(320, 224);  // (W, H)

    while (cap.isOpened()) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Preprocess the frame
        cv::Mat preprocessed_image = preprocess_image(frame, input_size, input_tensor);
        preprocessed_image = preprocessed_image.reshape(1, {1, input_size.height, input_size.width, 3});  // Add batch dimension

        // Set the input tensor
        memcpy(interpreter->typed_tensor<int8_t>(interpreter->inputs()[0]), preprocessed_image.data, preprocessed_image.total() * preprocessed_image.elemSize());

        // Run the model
        interpreter->Invoke();

        // Get the outputs as float32
        float* output_affinity = interpreter->typed_output_tensor<float>(0);
        float* output_heatmap = interpreter->typed_output_tensor<float>(1);

        // Extract keypoints
        int heatmap_height = output_heatmap_tensor->dims->data[1];
        int heatmap_width = output_heatmap_tensor->dims->data[2];
        int num_keypoints = output_heatmap_tensor->dims->data[3];
        cv::Mat heatmap(heatmap_height, heatmap_width, CV_32FC(num_keypoints), output_heatmap);
        auto [keypoints, probabilities] = extract_keypoints(heatmap);

        // Print keypoints and their probabilities
        for (size_t i = 0; i < keypoints_labels.size(); ++i) {
            std::cout << keypoints_labels[i] << ": " << keypoints[i] << ", probability: " << probabilities[i] << std::endl;
        }

        // Calculate scaling factors
        float scale_x = static_cast<float>(frame.cols) / input_size.width;
        float scale_y = static_cast<float>(frame.rows) / input_size.height;

        // Visualize the pose estimation on the frame
        visualize_pose(frame, keypoints, skeleton, scale_x, scale_y);

        // Write the frame to the output video
        video_writer.write(frame);

        // Display the frame
        cv::imshow("Pose Estimation", frame);

        // Break the loop if 'q' is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release the video and close all OpenCV windows
    cap.release();
    video_writer.release();
    cv::destroyAllWindows();

    return 0;
}
