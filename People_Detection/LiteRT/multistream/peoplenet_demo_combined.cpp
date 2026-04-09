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
#include <thread>
#include <mutex>

#include "absl/cleanup/cleanup.h"
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/delegate_loader.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/tflite_settings_json_parser.h"
#include "tensorflow/lite/interpreter_builder.h"

class DetectNetV2Model {
public:
    DetectNetV2Model(const std::string& model_path, int num_threads = 4) {
        // Load the model
        model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        if (!model) {
            throw std::runtime_error("Failed to load model");
        }
        // Setup Stable delegate
        using tflite::delegates::utils::LoadDelegateFromSharedLibrary;
        using tflite::delegates::utils::TfLiteSettingsJsonParser;
        constexpr char kSampleDelegatePath[] =
            "/usr/lib/libneuron_stable_delegate.so";
        constexpr char kSettingsPath[] =
            "/usr/share/label_image/stable_delegate_settings.json";

        // Load stable delegate.
        const TfLiteStableDelegate* stable_delegate =
             LoadDelegateFromSharedLibrary(kSampleDelegatePath);
        if (stable_delegate == nullptr || stable_delegate->delegate_plugin == nullptr)
             throw std::runtime_error("Failed to load stable delegate from library");

        // Load settings.
        TfLiteSettingsJsonParser parser;
        const tflite::TFLiteSettings* settings = parser.Parse(kSettingsPath);
        if (settings == nullptr)
             throw std::runtime_error("Failed to load json settings");

        // Create opaque delegate.
        TfLiteOpaqueDelegate* opaque_delegate =
            stable_delegate->delegate_plugin->create(settings);
        if (stable_delegate == nullptr)
             throw std::runtime_error("Failed to create opaque delegate");
        absl::Cleanup destory_opaque_delegate = [&] {
          stable_delegate->delegate_plugin->destroy(opaque_delegate);
        };

        // Build the interpreter
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*model, resolver)(&interpreter);
        if (!interpreter) {
            throw std::runtime_error("Failed to create interpreter");
        }

        // Add delegate to the interpreter
        if (interpreter->ModifyGraphWithDelegate(opaque_delegate) != kTfLiteOk)
            throw std::runtime_error("Failed to modify graph with Opaque delegate");

        // Set number of threads
        interpreter->SetNumThreads(num_threads);

        // Allocate tensor buffers
        if (interpreter->AllocateTensors() != kTfLiteOk) {
            throw std::runtime_error("Failed to allocate tensors");
        }

        // Get input and output tensor indices
        input_index = interpreter->inputs()[0];
        output_cov_index = interpreter->outputs()[0];
        output_bbox_index = interpreter->outputs()[1];
    }

    std::pair<std::vector<float>, std::vector<float>> predict(const cv::UMat& frame) {
        // Convert frame to int8 3-channel format
        cv::UMat input_frame;
        frame.convertTo(input_frame, CV_8SC3, 1, -128);

        // Set input tensor
        TfLiteTensor* input_tensor = interpreter->tensor(input_index);
        if (input_tensor == nullptr)
            throw std::runtime_error("Input tensor is nullptr");

        int8_t* input_data = input_tensor->data.int8;
        cv::Mat input_frame_mat = input_frame.getMat(cv::ACCESS_READ);
        std::memcpy(input_data, input_frame_mat.data, input_frame_mat.total() * input_frame_mat.elemSize());

        // Run inference
        if (interpreter->Invoke() != kTfLiteOk) {
            throw std::runtime_error("Failed to invoke interpreter");
        }

        // Get output tensors
        TfLiteTensor* output_cov_tensor = interpreter->tensor(output_cov_index);
        TfLiteTensor* output_bbox_tensor = interpreter->tensor(output_bbox_index);

        std::vector<float> cov(output_cov_tensor->data.f, output_cov_tensor->data.f + output_cov_tensor->bytes / sizeof(float));
        std::vector<float> bbox(output_bbox_tensor->data.f, output_bbox_tensor->data.f + output_bbox_tensor->bytes / sizeof(float));

        return {cov, bbox};
    }

    std::vector<int> get_output_shape(int output_index) {
        TfLiteTensor* output_tensor = interpreter->tensor(output_index);
        std::vector<int> shape(output_tensor->dims->size);
        for (int i = 0; i < output_tensor->dims->size; ++i) {
            shape[i] = output_tensor->dims->data[i];
        }
        return shape;
    }

    int get_output_cov_index() const {
        return output_cov_index;
    }

    int get_output_bbox_index() const {
        return output_bbox_index;
    }

private:
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    int input_index;
    int output_cov_index;
    int output_bbox_index;
};

class PostProcessingModel {
public:
    PostProcessingModel(int stride = 16, float scale = 35.0, float offset = 0.5)
        : stride(stride), scale(scale), offset(offset) {
        for (int i = 0; i < 60; ++i) {
            centers_x.push_back(i * stride + offset);
        }
        for (int i = 0; i < 34; ++i) {
            centers_y.push_back(i * stride + offset);
        }
    }

    std::vector<std::tuple<cv::Rect, int, float>> predict(const std::vector<float>& cov, const std::vector<float>& bbox, float threshold = 0.1, float nms_threshold = 0.05) {
        int num_detections = 2040;
        int num_classes = 3;
        int num_boxes = 12;

        std::vector<std::tuple<cv::Rect, int, float>> detections;
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        for (int i = 0; i < num_detections; ++i) {
            for (int c = 0; c < num_classes; ++c) {
                float confidence = cov[i * num_classes + c];
                if (confidence > threshold) {
                    int index = i * num_boxes + c * 4;
                    int w = i % 60;
                    int h = i / 60;
                    float x1 = centers_x[w] - scale * bbox[index + 0];
                    float y1 = centers_y[h] - scale * bbox[index + 1];
                    float x2 = centers_x[w] + scale * bbox[index + 2];
                    float y2 = centers_y[h] + scale * bbox[index + 3];
                    boxes.emplace_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
                    confidences.push_back(confidence);
                    class_ids.push_back(c);
                }
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, threshold, nms_threshold, indices);
        for (int idx : indices) {
            detections.emplace_back(boxes[idx], class_ids[idx], confidences[idx]);
        }

        return detections;
    }

private:
    int stride;
    float scale;
    float offset;
    std::vector<float> centers_x;
    std::vector<float> centers_y;
};

std::vector<std::string> load_labels(const std::string& label_file) {
    std::vector<std::string> labels;
    std::ifstream file(label_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open label file");
    }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        int index;
        std::string label;
        ss >> index >> label;
        if (ss.fail()) {
            throw std::runtime_error("Failed to parse label file");
        }
        labels.push_back(label);
    }
    return labels;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <label_file> <input_video1> <input_video2> <input_video3> <input_video4>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string label_file = argv[2];
    std::string input_video1 = argv[3];
    std::string input_video2 = argv[4];
    std::string input_video3 = argv[5];
    std::string input_video4 = argv[6];
    std::string output_video = "output.mp4";

    //std::string pipeline = "v4l2src device=/dev/video" + cam_id + " ! video/x-raw,width=640,height=480 ! v4l2convert ! video/x-raw,width=960,height=544,format=BGR ! appsink drop=true sync=false";
    std::string pipeline1 = "filesrc location=" + input_video1 + " ! parsebin ! v4l2h264dec ! v4l2convert ! video/x-raw,width=480,height=272,format=BGR ! appsink";
    std::string pipeline2 = "filesrc location=" + input_video2 + " ! parsebin ! v4l2h264dec ! v4l2convert ! video/x-raw,width=480,height=272,format=BGR ! appsink";
    std::string pipeline3 = "filesrc location=" + input_video3 + " ! parsebin ! v4l2h264dec ! v4l2convert ! video/x-raw,width=480,height=272,format=BGR ! appsink";
    std::string pipeline4 = "filesrc location=" + input_video4 + " ! parsebin ! v4l2h264dec ! v4l2convert ! video/x-raw,width=480,height=272,format=BGR ! appsink";

    cv::VideoCapture vid1(pipeline1, cv::CAP_GSTREAMER); // Open the specified camera using GStreamer pipeline
    if (!vid1.isOpened()) {
        std::cerr << "Error: Could not open video stream." << std::endl;
        return 1;
    }
    cv::VideoCapture vid2(pipeline2, cv::CAP_GSTREAMER); // Open the specified camera using GStreamer pipeline
    if (!vid2.isOpened()) {
        std::cerr << "Error: Could not open video stream." << std::endl;
        return 1;
    }
    cv::VideoCapture vid3(pipeline3, cv::CAP_GSTREAMER); // Open the specified camera using GStreamer pipeline
    if (!vid3.isOpened()) {
        std::cerr << "Error: Could not open video stream." << std::endl;
        return 1;
    }
    cv::VideoCapture vid4(pipeline4, cv::CAP_GSTREAMER); // Open the specified camera using GStreamer pipeline
    if (!vid4.isOpened()) {
        std::cerr << "Error: Could not open video stream." << std::endl;
        return 1;
    }

    // Get video properties
    int frame_width = static_cast<int>(vid1.get(cv::CAP_PROP_FRAME_WIDTH)*2);
    int frame_height = static_cast<int>(vid1.get(cv::CAP_PROP_FRAME_HEIGHT)*2);
    int fps = static_cast<int>(vid1.get(cv::CAP_PROP_FPS));

    // GStreamer pipeline for writing the output video
    std::string output_pipeline = "appsrc ! videoconvert ! queue ! v4l2h265enc extra-controls=\"controls,video_gop_size=10,video_bitrate=3600000,sequence_header_mode=1\" ! queue ! video/x-h265,level=(string)4,profile=main ! queue ! h265parse ! queue ! mp4mux ! queue ! filesink location=" + output_video;

    // Define the codec and create VideoWriter object using GStreamer pipeline
    cv::VideoWriter video_writer(output_pipeline, cv::CAP_GSTREAMER, 0, fps, cv::Size(frame_width, frame_height), true);

    if (!video_writer.isOpened()) {
        std::cerr << "Error: Could not open the output video file for write." << std::endl;
        return 1;
    }

    try {
        // Create an instance of the model
        DetectNetV2Model model(model_path);

        PostProcessingModel post_processor;
        std::vector<std::string> labels = load_labels(label_file);

        std::vector<cv::Scalar> colors = {cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255)};
        int min_height = 20;
        int frame_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (true) {
            cv::UMat frame1, frame2, frame3, frame4;
            bool ret1 = vid1.read(frame1);
            bool ret2 = vid2.read(frame2);
            bool ret3 = vid3.read(frame3);
            bool ret4 = vid4.read(frame4);

            if (!ret1 || !ret2 || !ret3 || !ret4) {
                break;
            }

            // Combine the frames into a single input frame
            cv::UMat top, bottom, combined;
            cv::hconcat(frame1, frame2, top);
            cv::hconcat(frame3, frame4, bottom);
            cv::vconcat(top, bottom, combined);

            // Perform inference on the combined frame
            auto [cov, bbox] = model.predict(combined);
            auto detections = post_processor.predict(cov, bbox);

            // Draw detections on the combined frame
            for (const auto& detection : detections) {
                const auto& box = std::get<0>(detection);
                int class_id = std::get<1>(detection);
                float confidence = std::get<2>(detection);
                cv::rectangle(combined, box, colors[class_id % colors.size()], 2);
            }

            // Calculate FPS
            frame_count++;
            auto current_time = std::chrono::high_resolution_clock::now();
            double elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            double fps = frame_count / elapsed_seconds;

            // Display FPS on the combined frame
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << fps;
            std::string fps_text = "FPS: " + oss.str();
            cv::putText(combined, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);

            // Write the frame to the output video file
            video_writer.write(combined);

            // Display the combined frame
            cv::imshow("DetectNet V2: " + model_path, combined);

            if (cv::waitKey(1) == 'q') {
                break;
            }
        }

        vid1.release();
        vid2.release();
        vid3.release();
        vid4.release();
        video_writer.release();
        cv::destroyAllWindows();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
