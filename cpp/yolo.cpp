#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip> // For std::fixed and std::setprecision
#include <chrono>  // For FPS calculation

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudacodec.hpp> // For GPU VideoReader
#include <opencv2/cudaarithm.hpp> // For GpuMat operations
#include <opencv2/cudaimgproc.hpp> // For cv::cuda::cvtColor

// YOLO Model Constants (adjust for your specific model if different)
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.25;       // General score threshold to consider a detection before NMS
const float NMS_THRESHOLD = 0.45;         // Non-Maximum Suppression IOU threshold

// Function to load class names from a file
std::vector<std::string> load_class_names(const std::string& path) {
    std::vector<std::string> class_names;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "Error: Failed to open class names file: " << path << std::endl;
        return class_names;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        class_names.push_back(line);
    }
    return class_names;
}

// draw_label function is removed as it was for GUI display

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <video_file_path> <object_name_to_detect> [confidence_threshold (default 0.5)]" << std::endl;
        std::cerr << "Example: " << argv[0] << " myvideo.mp4 dog 0.6" << std::endl;
        return -1;
    }

    std::string video_file = argv[1];
    std::string target_object_name = argv[2];
    float user_confidence_threshold = 0.5f; // Default

    if (argc > 3) {
        try {
            user_confidence_threshold = std::stof(argv[3]);
            if (user_confidence_threshold < 0.0f || user_confidence_threshold > 1.0f) {
                std::cerr << "Warning: Confidence threshold should be between 0.0 and 1.0. Using " 
                          << std::fixed << std::setprecision(2) << user_confidence_threshold << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Invalid confidence threshold '" << argv[3] << "'. Using default 0.5." << std::endl;
            user_confidence_threshold = 0.5f;
        }
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Video File: " << video_file << std::endl;
    std::cout << "  Target Object: " << target_object_name << std::endl;
    std::cout << "  Confidence Threshold for Target: " << std::fixed << std::setprecision(2) << user_confidence_threshold << std::endl;
    std::cout << "  General Score Threshold for NMS: " << SCORE_THRESHOLD << std::endl;
    std::cout << "  NMS IoU Threshold: " << NMS_THRESHOLD << std::endl;

    std::string model_path = "yolov8n.onnx"; 
    std::string class_names_path = "coco.names";

    std::vector<std::string> class_names = load_class_names(class_names_path);
    if (class_names.empty()) {
        return -1; 
    }
    int num_classes = class_names.size();
    std::cout << "Loaded " << num_classes << " class names." << std::endl;

    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(model_path);
        std::cout << "Successfully loaded ONNX model: " << model_path << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading ONNX model: " << e.what() << std::endl;
        return -1;
    }

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    std::cout << "Using CUDA backend for DNN inference." << std::endl;

    cv::Ptr<cv::cudacodec::VideoReader> d_reader;
    try {
        d_reader = cv::cudacodec::createVideoReader(video_file);
        cv::cudacodec::FormatInfo formatInfo = d_reader->format();
        if (formatInfo.width == 0 || formatInfo.height == 0) { 
             std::cerr << "Error: Could not get valid format info from video file using GPU: " << video_file << std::endl;
             std::cerr << "This might mean the video format or codec is not supported by NVDEC, or the file path is incorrect." << std::endl;
             return -1;
        }
        std::cout << "Video opened via GPU: " << formatInfo.width << "x" << formatInfo.height << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "Error creating GPU VideoReader for " << video_file << ": " << e.what() << std::endl;
        return -1;
    }
    
    cv::cuda::GpuMat gpu_frame;
    cv::cuda::GpuMat gpu_frame_bgr; 
    cv::Mat cpu_frame_for_blob; 
    cv::Mat blob;
    // cv::Mat cpu_display_frame; // Removed, as no display is needed

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    long long frame_processed_count = 0;
    std::cout << "Starting video processing loop..." << std::endl;
    auto loop_start_time = std::chrono::high_resolution_clock::now(); 

    while (true) { 
        if (!d_reader->nextFrame(gpu_frame)) {
            // std::cout << "End of video stream or failed to grab frame." << std::endl; // Less verbose for server
            break;
        }
        if (gpu_frame.empty()) {
            // std::cout << "Empty frame received, possibly end of stream." << std::endl; // Less verbose for server
            continue; 
        }
        frame_processed_count++;

        if (gpu_frame.channels() == 4) {
            cv::cuda::cvtColor(gpu_frame, gpu_frame_bgr, cv::COLOR_BGRA2BGR); 
        } else if (gpu_frame.channels() == 1) {
            cv::cuda::cvtColor(gpu_frame, gpu_frame_bgr, cv::COLOR_GRAY2BGR);
        } else if (gpu_frame.channels() == 3) {
            gpu_frame_bgr = gpu_frame; 
        } else {
            std::cerr << "Frame " << frame_processed_count << ": Error - Unsupported number of channels in GPU frame: " << gpu_frame.channels() << std::endl;
            continue;
        }

        if (gpu_frame_bgr.empty()) {
             std::cerr << "Frame " << frame_processed_count << ": Error - Frame became empty after GPU channel conversion." << std::endl;
             continue;
        }
        
        gpu_frame_bgr.download(cpu_frame_for_blob); 

        if (cpu_frame_for_blob.empty()) { 
            std::cerr << "Frame " << frame_processed_count << ": Error - Failed to download GpuMat to Mat for blob creation, or frame is empty." << std::endl;
            continue; 
        }
        
        cv::dnn::blobFromImage(cpu_frame_for_blob, blob, 1.0/255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false, CV_32F);
        
        net.setInput(blob);

        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames()); 

        cv::Mat output_layer = outs[0]; 
        if (output_layer.dims < 3 || output_layer.size[0] != 1 || output_layer.size[1] != (4 + num_classes) || output_layer.size[2] == 0) {
            // No detections or unexpected output, just continue to next frame for server application
            if (frame_processed_count % 100 == 0) { // Print occasionally to show progress/issue
                 std::cout << "Frame " << frame_processed_count << ": No valid detections or unexpected model output." << std::endl;
            }
            continue; 
        }

        cv::Mat temp_mat = output_layer.reshape(1, output_layer.size[1]); 
        cv::Mat detections_transposed_channels; 
        cv::transpose(temp_mat, detections_transposed_channels); 

        class_ids.clear();
        confidences.clear();
        boxes.clear();

        float x_factor = (float)gpu_frame_bgr.cols / INPUT_WIDTH;
        float y_factor = (float)gpu_frame_bgr.rows / INPUT_HEIGHT;

        for (int i = 0; i < detections_transposed_channels.rows; ++i) { 
            cv::Mat detection_row = detections_transposed_channels.row(i); 
            float* data = (float*)detection_row.data; 
            
            cv::Mat class_scores_mat = detection_row.colRange(4, 4 + num_classes);
            cv::Point class_id_point;
            double max_class_score_double; 
            cv::minMaxLoc(class_scores_mat, 0, &max_class_score_double, 0, &class_id_point);
            float max_class_score = static_cast<float>(max_class_score_double);

            if (max_class_score > SCORE_THRESHOLD) {
                confidences.push_back(max_class_score);
                class_ids.push_back(class_id_point.x);

                float cx = data[0]; 
                float cy = data[1]; 
                float w = data[2];  
                float h = data[3];  

                int left = static_cast<int>((cx - w / 2) * x_factor);
                int top = static_cast<int>((cy - h / 2) * y_factor);
                int width = static_cast<int>(w * x_factor);
                int height = static_cast<int>(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        } 

        std::vector<int> nms_indices;
        if (!boxes.empty()) { 
             cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_indices);
        }
        
        // GUI related drawing and display are removed.
        // We will just print to console if the target object is found.
        for (int idx : nms_indices) {
            // cv::Rect box = boxes[idx]; // Not needed if not drawing
            int class_id = class_ids[idx];
            float confidence_val = confidences[idx];
            std::string class_name = (class_id >= 0 && class_id < (int)class_names.size()) ? class_names[class_id] : "Unknown";

            if (class_name == target_object_name && confidence_val >= user_confidence_threshold) {
                cv::Rect box = boxes[idx]; // Get box only if it's the target
                std::cout << "Frame " << frame_processed_count << ": Detected '" << target_object_name << "' with confidence "
                          << std::fixed << std::setprecision(2) << confidence_val
                          << " at [" << box.x << ", " << box.y << ", " << box.width << ", " << box.height << "]" << std::endl;
                // Here you could trigger saving the frame (e.g., cpu_frame_for_blob) to disk asynchronously
            }
        }
        // cv::waitKey(1) is removed as it's for GUI interaction.
        // For a server app, the loop will run as fast as possible.
        // If you need to control processing speed or exit, other mechanisms would be needed (e.g. signal handling).
    } 

    auto overall_end_time = std::chrono::high_resolution_clock::now();
    auto total_processing_duration = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end_time - loop_start_time);
    double avg_fps = 0.0;
    if (total_processing_duration.count() > 0 && frame_processed_count > 0) {
        avg_fps = static_cast<double>(frame_processed_count) / (total_processing_duration.count() / 1000.0);
    }
    
    std::cout << "Processing finished." << std::endl;
    std::cout << "Total frames processed: " << frame_processed_count << std::endl;
    std::cout << "Total processing time: " << std::fixed << std::setprecision(2) << total_processing_duration.count() / 1000.0 << " seconds" << std::endl;
    std::cout << "Average FPS (overall): " << std::fixed << std::setprecision(2) << avg_fps << std::endl;

    // cv::destroyAllWindows(); // Removed
    return 0;
}
