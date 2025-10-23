#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <memory>
#include <iomanip>
#include <sstream>
#include <tuple>
#include <stdexcept>
#include "onnxruntime/core/providers/neuron/neuron_provider_factory.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

// ONNX Model Runner with NeuronExecutionProvider support
class DetectNetV2ONNXModel {
public:
    DetectNetV2ONNXModel(const std::string& model_path, int num_threads = 4,
                         const std::string& backend = "cpu",
                         const std::string& min_num_in_group = "1")
        : env(ORT_LOGGING_LEVEL_WARNING, "DetectNetV2ONNX") {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        // Enable NeuronExecutionProvider if requested
        if (backend == "neuron") {
            const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
            std::vector<const char*> options_keys = {"NEURON_FLAG_USE_FP16", "NEURON_FLAG_MIN_GROUP_SIZE"};
            std::vector<const char*> options_values = {"1", min_num_in_group.c_str()};
            if (g_ort->SessionOptionsAppendExecutionProvider(
                    session_options,
                    "Neuron",
                    options_keys.data(),
                    options_values.data(),
                    options_keys.size()) != 0) {
                std::cout << "Failed to append Neuron execution provider" << std::endl;
                throw std::runtime_error("Failed to append Neuron execution provider");
            }
            std::cout << "Using NeuronExecutionProvider" << std::endl;
        } else if (backend == "cpu") {
            std::cout << "Using CPUExecutionProvider (default)" << std::endl;
            // No need to append, it's the default
        } else {
            std::cout << "Not supported backend: " << backend << std::endl;
            throw std::runtime_error("Not supported backend: " + backend);
        }

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        input_name = session->GetInputNameAllocated(0, allocator).get();
        output_cov_name = session->GetOutputNameAllocated(0, allocator).get();
        output_bbox_name = session->GetOutputNameAllocated(1, allocator).get();
    }

    std::pair<std::vector<float>, std::vector<float>> predict(const cv::Mat& frame) {
        // Preprocess: Resize to 960x544, convert to FP32, normalize, NCHW
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(960, 544));
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

        // Convert HWC to NCHW
        std::vector<float> input_tensor(1 * 3 * 544 * 960);
        for (int y = 0; y < 544; ++y) {
            for (int x = 0; x < 960; ++x) {
                cv::Vec3f pixel = float_img.at<cv::Vec3f>(y, x);
                for (int c = 0; c < 3; ++c) {
                    input_tensor[c * 544 * 960 + y * 960 + x] = pixel[c];
                }
            }
        }

        std::array<int64_t, 4> input_shape = {1, 3, 544, 960};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor.data(), input_tensor.size(),
            input_shape.data(), input_shape.size());

        const char* input_name_cstr = input_name.c_str();
        const char* output_names[] = {output_cov_name.c_str(), output_bbox_name.c_str()};

        auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                           &input_name_cstr, &input_tensor_ort, 1,
                                           output_names, 2);

        float* cov_ptr = output_tensors[0].GetTensorMutableData<float>();
        float* bbox_ptr = output_tensors[1].GetTensorMutableData<float>();

        std::vector<float> cov(cov_ptr, cov_ptr + 1 * 3 * 34 * 60);
        std::vector<float> bbox(bbox_ptr, bbox_ptr + 1 * 12 * 34 * 60);

        return {cov, bbox};
    }

private:
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    std::string input_name;
    std::string output_cov_name;
    std::string output_bbox_name;
};

// Post-processing (unchanged from your original)
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

    std::vector<std::tuple<cv::Rect, int, float>> predict(const std::vector<float>& cov, const std::vector<float>& bbox, float threshold = 0.05, float nms_threshold = 0.01) {
        int num_detections = 2040; // 34*60
        int num_classes = 3;
        int num_boxes = 12;

        std::vector<std::tuple<cv::Rect, int, float>> detections;
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        for (int i = 0; i < num_detections; ++i) {
            int w = i % 60;
            int h = i / 60;
            for (int c = 0; c < num_classes; ++c) {
                float confidence = cov[c * 34 * 60 + i];
                if (confidence > threshold) {
                    int index = (c * 4) * 34 * 60 + i;
                    float x1 = centers_x[w] - scale * bbox[index + 0 * 34 * 60];
                    float y1 = centers_y[h] - scale * bbox[index + 1 * 34 * 60];
                    float x2 = centers_x[w] + scale * bbox[index + 2 * 34 * 60];
                    float y2 = centers_y[h] + scale * bbox[index + 3 * 34 * 60];
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
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <label_file> <input_video> <output_video> [backend] [min_group_size]" << std::endl;
        std::cerr << "  backend: 'cpu' or 'neuron'" << std::endl;
        std::cerr << "  min_group_size: optional, default is '1'" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string label_file = argv[2];
    std::string input_video = argv[3];
    std::string output_video = argv[4];
    std::string backend = (argc > 5) ? argv[5] : "cpu";
    std::string min_num_in_group = (argc > 6) ? argv[6] : "1";

    // GStreamer pipeline for reading the input video and resizing to 1080p
    std::string input_pipeline = "filesrc location=" + input_video + " ! parsebin ! v4l2h264dec ! v4l2convert ! video/x-raw,width=1920,height=1080,format=BGR ! appsink";

    cv::VideoCapture vid(input_pipeline, cv::CAP_GSTREAMER); // Open the specified video file using GStreamer pipeline
    if (!vid.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return 1;
    }

    // Get video properties
    int frame_width = static_cast<int>(vid.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(vid.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(vid.get(cv::CAP_PROP_FPS));

    // GStreamer pipeline for writing the output video
    std::string output_pipeline = "appsrc ! videoconvert ! queue ! v4l2h265enc extra-controls=\"controls,video_gop_size=10,video_bitrate=3600000,sequence_header_mode=1\" ! queue ! video/x-h265,level=(string)4,profile=main ! queue ! h265parse ! queue ! mp4mux ! queue ! filesink location=" + output_video; 

    // Define the codec and create VideoWriter object using GStreamer pipeline
    cv::VideoWriter video_writer(output_pipeline, cv::CAP_GSTREAMER, 0, fps, cv::Size(frame_width, frame_height), true);

    if (!video_writer.isOpened()) {
        std::cerr << "Error: Could not open the output video file for write." << std::endl;
        return 1;
    }

    try {
        DetectNetV2ONNXModel model(model_path, 4, backend, min_num_in_group);
        PostProcessingModel post_processor;
        std::vector<std::string> labels = load_labels(label_file);

        std::vector<cv::Scalar> colors = {cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255)};
        int frame_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (true) {
            cv::Mat frame;
            bool ret = vid.read(frame);

            if (!ret) {
                break;
            }

            // Resize frame to 960x544 for model input
            cv::Mat resized_frame;
            cv::resize(frame, resized_frame, cv::Size(960, 544));
            cv::cvtColor(resized_frame, resized_frame, cv::COLOR_BGR2RGB);

            auto predict_start = std::chrono::high_resolution_clock::now();
            auto [cov, bbox] = model.predict(resized_frame);
            auto predict_end = std::chrono::high_resolution_clock::now();
            auto predict_duration = std::chrono::duration_cast<std::chrono::milliseconds>(predict_end - predict_start).count();

            auto post_process_start = std::chrono::high_resolution_clock::now();
            std::vector<std::tuple<cv::Rect, int, float>> detections = post_processor.predict(cov, bbox);
            auto post_process_end = std::chrono::high_resolution_clock::now();

            // Convert frame to UMat for GPU processing
            if (!cv::ocl::haveOpenCL()) {
                std::cout << "OpenCL is not available..." << std::endl;
            }
            cv::UMat uframe;
            frame.copyTo(uframe);

            for (const auto& detection : detections) {
                const auto& box = std::get<0>(detection);
                int class_id = std::get<1>(detection);
                float confidence = std::get<2>(detection);

                // Scale the box coordinates back to the original 1080p frame size
                cv::Rect scaled_box(
                    static_cast<int>(box.x * (frame_width / 960.0)),
                    static_cast<int>(box.y * (frame_height / 544.0)),
                    static_cast<int>(box.width * (frame_width / 960.0)),
                    static_cast<int>(box.height * (frame_height / 544.0))
                );

                // Draw rectangle on GPU
                cv::rectangle(uframe, scaled_box, colors[class_id % colors.size()], 2, cv::LINE_8);
                // Optionally draw label text:
                // std::string label = labels[class_id] + ": " + std::to_string(confidence);
                // cv::putText(uframe, label, cv::Point(scaled_box.x, scaled_box.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
            }

            frame_count++;
            auto current_time = std::chrono::high_resolution_clock::now();
            double elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            double fps = frame_count / std::max(1.0, elapsed_seconds);
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << fps;
            std::string fps_text = "FPS: " + oss.str();

            std::ostringstream predict_oss;
            predict_oss << predict_duration << " ms";
            std::string predict_text = "Inference: " + predict_oss.str();

            // Optionally display FPS and inference time
            // cv::putText(uframe, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
            // cv::putText(uframe, predict_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

            // Convert UMat back to Mat for display and saving
            uframe.copyTo(frame);

            // Write the frame to the output video file
            video_writer.write(frame);

            cv::imshow("DetectNet V2 ONNX: " + model_path, frame);

            if (cv::waitKey(1) == 'q') {
                break;
            }
        }

        vid.release();
        video_writer.release();
        cv::destroyAllWindows();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}