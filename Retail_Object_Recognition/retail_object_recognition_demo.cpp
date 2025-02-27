#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>
#include "absl/cleanup/cleanup.h"
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/delegate_loader.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/tflite_settings_json_parser.h"
#include "tensorflow/lite/interpreter_builder.h"
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/string_util.h>
#include <tensorflow/lite/kernels/register.h>

// Function to preprocess the image
cv::Mat preprocessImage(const cv::Mat& image) {
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(224, 224));

    // Convert the image to float and normalize to [0, 1]
    resizedImage.convertTo(resizedImage, CV_32FC3, 1.0 / 255.0);

    // Normalize to [-1, 1]
    resizedImage = (resizedImage - 0.5) * 2;

    return resizedImage;
}

// Struct to hold logistic regression parameters
struct LogisticRegressionParams {
    std::vector<float> coefficients;
    float intercept;
};

// Function to load the CSV file
LogisticRegressionParams loadCSV(const std::string& csvPath) {
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the CSV file!" << std::endl;
        exit(EXIT_FAILURE);
    }

    LogisticRegressionParams params;
    std::string line;

    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        // Skip the label
        std::getline(ss, value, ',');

        // Read the coefficient
        std::getline(ss, value, ',');
        try {
            params.coefficients.push_back(std::stof(value));
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid coefficient value: " << value << std::endl;
            exit(EXIT_FAILURE);
        }

        // Read the intercept
        std::getline(ss, value, ',');
        try {
            params.intercept = std::stof(value);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid intercept value: " << value << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    file.close();
    return params;
}

// Function to compute the decision function
float computeDecisionFunction(const std::vector<float>& features, const LogisticRegressionParams& params) {
    float decisionFunction = params.intercept;
    for (size_t i = 0; i < features.size(); ++i) {
        decisionFunction += features[i] * params.coefficients[i];
    }
    return decisionFunction;
}

// Function to make a prediction and return the probability
float makePrediction(float decisionFunction) {
    return 1 / (1 + std::exp(-decisionFunction));
}

// Function to display the frame with the predicted label and other information
void displayFrameWithLabel(cv::Mat& frame, const std::string& label, int cokeCount, int pepsiCount, float totalCost) {
    // Define the text to be displayed
    std::string text = "Class: " + label;
    std::string cokeText = "Number of coke: " + std::to_string(cokeCount) + " x $1";
    std::string pepsiText = "Number of pepsi: " + std::to_string(pepsiCount) + " x $1.5";
    std::string costText = "Total Cost: $" + std::to_string(totalCost).substr(0, std::to_string(totalCost).find(".") + 3);

    // Define the font, scale, color, and thickness
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1;
    int thickness = 2;
    cv::Scalar color(0, 0, 255); // Red color

    // Put the text on the frame
    //cv::putText(frame, text, cv::Point(10, 30), font, fontScale, color, thickness);
    cv::putText(frame, cokeText, cv::Point(10, 30), font, fontScale, color, thickness);
    cv::putText(frame, pepsiText, cv::Point(10, 80), font, fontScale, color, thickness);
    cv::putText(frame, costText, cv::Point(10, 130), font, fontScale, color, thickness);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <csv_path> <output_video>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string modelPath = argv[1];
    std::string csvPath = argv[2];
    //int cam_id = std::stoi(argv[3]);
    std::string output_video = argv[3];

    // Load the CSV file
    LogisticRegressionParams params = loadCSV(csvPath);

    // Load TFLite Model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        exit(EXIT_FAILURE);
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

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to construct interpreter" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (interpreter->ModifyGraphWithDelegate(opaque_delegate) != kTfLiteOk)
        throw std::runtime_error("Failed to modify graph with Opaque delegate");

    // Allocate tensor buffers
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        exit(EXIT_FAILURE);
    }

    // GStreamer pipeline for camera input
    //std::string gstPipeline = "v4l2src device=/dev/video" + cam_id + " ! video/x-raw,width=1280,height=720,format=BGR ! videoconvert ! video/x-raw,width=1280,height=720,format=BGR ! appsink drop=true sync=false";
    cv::VideoCapture cap(5);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera!" << std::endl;
        return EXIT_FAILURE;
    }

    // Get video properties

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));


    // GStreamer pipeline for video output
    //std::string output_pipeline = "appsrc ! v4l2h265enc extra-controls=\"controls,video_gop_size=10,video_bitrate=3600000,sequence_header_mode=1\" ! queue ! video/x-h265,level=(string)4,profile=main ! queue ! h265parse ! queue ! mp4mux ! queue ! filesink location=" + output_video; 
    //std::string output_pipeline = "appsrc ! videoconvert ! v4l2h264enc ! mp4mux ! filesink location=" + output_video;

    // Define the codec and create VideoWriter object using GStreamer pipeline
    //cv::VideoWriter video_writer(output_pipeline, cv::CAP_GSTREAMER, 0, fps, cv::Size(frame_width, frame_height), true);
    cv::VideoWriter video_writer(output_video, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height), true);

    if (!video_writer.isOpened()) {
        std::cerr << "Error: Could not open the video writer!" << std::endl;
        return EXIT_FAILURE;
    }

    int cokeCount = 0;
    int pepsiCount = 0;
    float totalCost = 0.0;
    bool cokeDetected = false;
    bool pepsiDetected = false;

    // State machine variables
    enum State { NO_DETECTION, POSSIBLE_COKE, CONFIRMED_COKE, POSSIBLE_PEPSI, CONFIRMED_PEPSI };
    State currentState = NO_DETECTION;
    int frameCounter = 0;
    const int confirmationFrames = 5; // Number of consecutive frames to confirm detection

    cv::Mat frame;
    while (cap.read(frame)) {
        // Preprocess the frame
        cv::Mat inputImage = preprocessImage(frame);

        float* input = interpreter->typed_input_tensor<float>(0);

        // Copy the image data to the input tensor
        std::memcpy(input, inputImage.data, inputImage.total() * inputImage.elemSize());

        // Run the model
        auto start = std::chrono::high_resolution_clock::now();
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Failed to invoke tflite" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> inferenceTime = end - start;
        std::cout << "Inference time: " << inferenceTime.count() << " seconds" << std::endl;

        // Get output tensor
        float* output = interpreter->typed_output_tensor<float>(0);
        std::vector<float> outputVector(output, output + 2048);

        // Run the TFLite model
        std::vector<float> features = outputVector;

        // Compute the decision function
        float decisionFunction = computeDecisionFunction(features, params);

        // Make a prediction and get the probability
        float probability = makePrediction(decisionFunction);
        std::cout << "Probability: " << probability << std::endl;

        // State machine logic
        std::string currentLabel = "None";
        switch (currentState) {
            case NO_DETECTION:
                if (probability > 0.82) {
                    currentState = POSSIBLE_COKE;
                    frameCounter = 1;
                } else if (probability < 0.16) {
                    currentState = POSSIBLE_PEPSI;
                    frameCounter = 1;
                }
                break;
            case POSSIBLE_COKE:
                if (probability > 0.82) {
                    frameCounter++;
                    if (frameCounter >= confirmationFrames) {
                        currentState = CONFIRMED_COKE;
                        cokeCount++;
                        totalCost += 1.0;
                        std::cout << "Number of coke: " << cokeCount << std::endl;
                    }
                } else {
                    currentState = NO_DETECTION;
                }
                break;
            case CONFIRMED_COKE:
                if (probability <= 0.82) {
                    currentState = NO_DETECTION;
                }
                currentLabel = "coca-cola";
                break;
            case POSSIBLE_PEPSI:
                if (probability < 0.16) {
                    frameCounter++;
                    if (frameCounter >= confirmationFrames) {
                        currentState = CONFIRMED_PEPSI;
                        pepsiCount++;
                        totalCost += 1.5;
                        std::cout << "Number of pepsi: " << pepsiCount << std::endl;
                    }
                } else {
                    currentState = NO_DETECTION;
                }
                break;
            case CONFIRMED_PEPSI:
                if (probability >= 0.16) {
                    currentState = NO_DETECTION;
                }
                currentLabel = "pepsi";
                break;
        }

        // Display the frame with the label and other information
        displayFrameWithLabel(frame, currentLabel, cokeCount, pepsiCount, totalCost);

        // Write the frame to the video file
        video_writer.write(frame);

        // Show the frame
        cv::imshow("Live Camera", frame);
        if (cv::waitKey(30) >= 0) break; // Exit if any key is pressed
    }

    cap.release();
    video_writer.release();
    cv::destroyAllWindows();

    std::cout << "Total cost: $" << totalCost << std::endl;

    return 0;
}
