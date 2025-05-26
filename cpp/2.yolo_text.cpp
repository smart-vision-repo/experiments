#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip> // For std::fixed and std::setprecision
#include <chrono>  // For FPS calculation and time-based skipping
#include <cmath>   // For fmod and floor used in time calculation

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudacodec.hpp> // For GPU VideoReader
#include <opencv2/cudaarithm.hpp> // For GpuMat operations
#include <opencv2/cudaimgproc.hpp> // For cv::cuda::cvtColor

// YOLO Model Constants
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.25; // General score threshold to consider a detection before NMS
const float NMS_THRESHOLD = 0.45;   // Non-Maximum Suppression IOU threshold
const int PROCESSING_INTERVAL_MS = 1000;

#define DEBUG_PRINT_ALL_DETECTIONS 0 // 设置为 1 来打印所有NMS后的检测结果, 设置为 0 关闭

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

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <video_file_path> <object_name_to_detect> [confidence_threshold (default 0.5)]" << std::endl;
        return -1;
    }

    std::string video_file = argv[1];
    std::string target_object_name = argv[2];
    float user_confidence_threshold = 0.5f;

    if (argc > 3) {
        try {
            user_confidence_threshold = std::stof(argv[3]);
        } catch (const std::exception& e) {
            user_confidence_threshold = 0.5f;
            std::cerr << "Warning: Invalid confidence threshold '" << argv[3] << "'. Using default 0.5." << std::endl;
        }
    }

    // std::string model_path = "yolo11x.onnx"; 
    std::string model_path = "yolov8n.onnx"; 
    std::string class_names_path = "coco.names";

    std::vector<std::string> class_names = load_class_names(class_names_path);
    if (class_names.empty()) return -1; 
    int num_classes = class_names.size();
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Video File: " << video_file << std::endl;
    std::cout << "  Target Object: " << target_object_name << std::endl;
    std::cout << "  Confidence Threshold for Target: " << std::fixed << std::setprecision(2) << user_confidence_threshold << std::endl;
    std::cout << "  Model Path: " << model_path << std::endl;
    std::cout << "  Class Names Path: " << class_names_path << " (Loaded " << num_classes << " classes)" << std::endl;
    std::cout << "  Target Processing Interval: approx. " << PROCESSING_INTERVAL_MS << " ms" << std::endl;

    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(model_path);
        std::cout << "Successfully loaded ONNX model." << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading ONNX model: " << e.what() << std::endl;
        return -1;
    }

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    std::cout << "Using CUDA backend for DNN inference." << std::endl;

    cv::Ptr<cv::cudacodec::VideoReader> d_reader;
    cv::cudacodec::FormatInfo formatInfo; // Declare here to access fps later
    try {
        d_reader = cv::cudacodec::createVideoReader(video_file);
        formatInfo = d_reader->format();
        if (formatInfo.width == 0 || formatInfo.height == 0) { 
             std::cerr << "Error: Could not get valid format info from video file using GPU: " << video_file << std::endl;
             return -1;
        }
        std::cout << "Video opened via GPU: " << formatInfo.width << "x" << formatInfo.height << std::endl;
        if (formatInfo.fps > 0) {
            std::cout << "Video FPS (from container): " << formatInfo.fps << std::endl;
        } else {
            std::cout << "Warning: Video FPS from container is 0 or invalid. Video time in output will not be calculated." << std::endl;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Error creating GPU VideoReader for " << video_file << ": " << e.what() << std::endl;
        return -1;
    }
    
    cv::cuda::GpuMat gpu_frame;
    cv::cuda::GpuMat gpu_frame_bgr_for_processing; 
    cv::Mat cpu_frame_for_blob; 
    cv::Mat blob;

    std::vector<int> class_ids_vec; 
    std::vector<float> confidences_vec; 
    std::vector<cv::Rect> boxes_vec; 

    long long yolo_processed_frame_count = 0;
    long long total_frames_read = 0; // Renamed from frame_processed_count in previous version for clarity
    
    std::chrono::microseconds total_gpu_conversion_time_us(0);
    std::chrono::microseconds total_download_to_cpu_time_us(0); 
    std::chrono::microseconds total_blob_from_cpumat_time_us(0); 
    std::chrono::microseconds total_set_input_time_us(0);
    std::chrono::microseconds total_forward_time_us(0);
    std::chrono::microseconds total_post_processing_time_us(0);
    std::chrono::microseconds cumulative_yolo_pipeline_time_us(0);
    std::chrono::microseconds total_videoreader_nextframe_time_us(0);


    std::cout << "Starting video processing loop..." << std::endl;
    auto overall_program_start_time = std::chrono::high_resolution_clock::now();
    auto last_yolo_processed_time = std::chrono::high_resolution_clock::now() - std::chrono::milliseconds(PROCESSING_INTERVAL_MS); 
    std::chrono::high_resolution_clock::time_point stage_start_time, stage_end_time, frame_read_start_time, frame_read_end_time;

    while (true) { 
        frame_read_start_time = std::chrono::high_resolution_clock::now();
        if (!d_reader->nextFrame(gpu_frame)) {
            frame_read_end_time = std::chrono::high_resolution_clock::now();
            total_videoreader_nextframe_time_us += std::chrono::duration_cast<std::chrono::microseconds>(frame_read_end_time - frame_read_start_time);
            break; 
        }
        frame_read_end_time = std::chrono::high_resolution_clock::now();
        total_videoreader_nextframe_time_us += std::chrono::duration_cast<std::chrono::microseconds>(frame_read_end_time - frame_read_start_time);

        if (gpu_frame.empty()) {
            continue; 
        }
        total_frames_read++; // This now correctly counts frames read

        auto current_frame_time = std::chrono::high_resolution_clock::now();
        auto time_since_last_yolo = std::chrono::duration_cast<std::chrono::milliseconds>(current_frame_time - last_yolo_processed_time);

        if (time_since_last_yolo.count() >= PROCESSING_INTERVAL_MS) {
            auto single_pipeline_start_time = std::chrono::high_resolution_clock::now();
            last_yolo_processed_time = current_frame_time; 
            yolo_processed_frame_count++;

            stage_start_time = std::chrono::high_resolution_clock::now();
            if (gpu_frame.channels() == 4) {
                cv::cuda::cvtColor(gpu_frame, gpu_frame_bgr_for_processing, cv::COLOR_BGRA2BGR); 
            } else if (gpu_frame.channels() == 1) {
                cv::cuda::cvtColor(gpu_frame, gpu_frame_bgr_for_processing, cv::COLOR_GRAY2BGR);
            } else if (gpu_frame.channels() == 3) {
                gpu_frame_bgr_for_processing = gpu_frame; 
            } else {
                std::cerr << "Video Frame " << total_frames_read << ": Error - Unsupported GPU frame channels: " << gpu_frame.channels() << std::endl;
                continue;
            }
            stage_end_time = std::chrono::high_resolution_clock::now();
            total_gpu_conversion_time_us += std::chrono::duration_cast<std::chrono::microseconds>(stage_end_time - stage_start_time);

            if (gpu_frame_bgr_for_processing.empty()) {
                 std::cerr << "Video Frame " << total_frames_read << ": Error - Frame empty after GPU conversion." << std::endl;
                 continue;
            }
            
            stage_start_time = std::chrono::high_resolution_clock::now();
            gpu_frame_bgr_for_processing.download(cpu_frame_for_blob); 
            stage_end_time = std::chrono::high_resolution_clock::now();
            total_download_to_cpu_time_us += std::chrono::duration_cast<std::chrono::microseconds>(stage_end_time - stage_start_time);

            if (cpu_frame_for_blob.empty()) { 
                std::cerr << "Video Frame " << total_frames_read << ": Error - Failed to download GpuMat to Mat for blob creation." << std::endl;
                continue; 
            }
            
            stage_start_time = std::chrono::high_resolution_clock::now();
            cv::dnn::blobFromImage(cpu_frame_for_blob, blob, 1.0/255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false, CV_32F);
            stage_end_time = std::chrono::high_resolution_clock::now();
            total_blob_from_cpumat_time_us += std::chrono::duration_cast<std::chrono::microseconds>(stage_end_time - stage_start_time);
            
            stage_start_time = std::chrono::high_resolution_clock::now();
            net.setInput(blob); 
            stage_end_time = std::chrono::high_resolution_clock::now();
            total_set_input_time_us += std::chrono::duration_cast<std::chrono::microseconds>(stage_end_time - stage_start_time);

            std::vector<cv::Mat> outs;
            stage_start_time = std::chrono::high_resolution_clock::now();
            net.forward(outs, net.getUnconnectedOutLayersNames()); 
            stage_end_time = std::chrono::high_resolution_clock::now();
            total_forward_time_us += std::chrono::duration_cast<std::chrono::microseconds>(stage_end_time - stage_start_time);

            stage_start_time = std::chrono::high_resolution_clock::now();
            cv::Mat output_layer = outs[0]; 
            if (output_layer.dims < 3 || output_layer.size[0] != 1 || output_layer.size[1] != (4 + num_classes) || output_layer.size[2] == 0) {
                if (yolo_processed_frame_count % 10 == 0) { // Less frequent warning for server
                    std::cout << "YOLO Frame " << yolo_processed_frame_count << " (Video Frame " << total_frames_read 
                              << "): No valid detections or unexpected model output." << std::endl;
                }
                auto single_pipeline_end_time_early = std::chrono::high_resolution_clock::now();
                cumulative_yolo_pipeline_time_us += std::chrono::duration_cast<std::chrono::microseconds>(single_pipeline_end_time_early - single_pipeline_start_time);
                continue; 
            }

            cv::Mat temp_mat = output_layer.reshape(1, output_layer.size[1]); 
            cv::Mat detections_transposed_channels; 
            cv::transpose(temp_mat, detections_transposed_channels); 

            class_ids_vec.clear();
            confidences_vec.clear();
            boxes_vec.clear();

            float x_factor = (float)gpu_frame_bgr_for_processing.cols / INPUT_WIDTH;
            float y_factor = (float)gpu_frame_bgr_for_processing.rows / INPUT_HEIGHT;

            for (int i = 0; i < detections_transposed_channels.rows; ++i) { 
                cv::Mat detection_row = detections_transposed_channels.row(i); 
                float* data_ptr = (float*)detection_row.data; 
                
                cv::Mat class_scores_mat = detection_row.colRange(4, 4 + num_classes);
                cv::Point class_id_point;
                double max_class_score_double; 
                cv::minMaxLoc(class_scores_mat, 0, &max_class_score_double, 0, &class_id_point);
                float max_class_score = static_cast<float>(max_class_score_double);

                if (max_class_score > SCORE_THRESHOLD) {
                    confidences_vec.push_back(max_class_score);
                    class_ids_vec.push_back(class_id_point.x);

                    float cx = data_ptr[0]; float cy = data_ptr[1]; float w = data_ptr[2]; float h = data_ptr[3];  
                    int left = static_cast<int>((cx - w / 2) * x_factor);
                    int top = static_cast<int>((cy - h / 2) * y_factor);
                    int width = static_cast<int>(w * x_factor);
                    int height = static_cast<int>(h * y_factor);
                    boxes_vec.push_back(cv::Rect(left, top, width, height));
                }
            } 

            std::vector<int> nms_indices;
            if (!boxes_vec.empty()) { 
                 cv::dnn::NMSBoxes(boxes_vec, confidences_vec, SCORE_THRESHOLD, NMS_THRESHOLD, nms_indices);
            }
            
            // Debug print all NMS'd detections if enabled
            #if DEBUG_PRINT_ALL_DETECTIONS
            if (!nms_indices.empty()) {
                std::cout << "--- All Detections in YOLO Frame " << yolo_processed_frame_count 
                          << " (Video Frame " << total_frames_read << ") ---" << std::endl;
                for (int temp_idx : nms_indices) {
                    std::string temp_class_name = (class_ids_vec[temp_idx] >= 0 && class_ids_vec[temp_idx] < (int)class_names.size()) ? class_names[class_ids_vec[temp_idx]] : "Unknown";
                    std::cout << "  Found: '" << temp_class_name 
                              << "', Conf: " << std::fixed << std::setprecision(2) << confidences_vec[temp_idx]
                              << " at [" << boxes_vec[temp_idx].x << ", " << boxes_vec[temp_idx].y << ", " << boxes_vec[temp_idx].width << ", " << boxes_vec[temp_idx].height << "]"
                              << std::endl;
                }
                 std::cout << "----------------------------------------------------" << std::endl;
            } else if (yolo_processed_frame_count % 10 == 0) { // Print occasionally if no detections
                std::cout << "YOLO Frame " << yolo_processed_frame_count << " (Video Frame " << total_frames_read 
                          << "): No objects detected after NMS." << std::endl;
            }
            #endif
            
            for (int idx : nms_indices) {
                int class_id = class_ids_vec[idx];
                float confidence_val = confidences_vec[idx];
                std::string class_name_str = (class_id >= 0 && class_id < (int)class_names.size()) ? class_names[class_id] : "Unknown"; // Renamed

                if (class_name_str == target_object_name && confidence_val >= user_confidence_threshold) {
                    cv::Rect box = boxes_vec[idx]; 
                    std::cout << "YOLO Frame " << yolo_processed_frame_count << " (Video Frame " << total_frames_read 
                              << "): Detected '" << target_object_name << "' with confidence "
                              << std::fixed << std::setprecision(2) << confidence_val
                              << " at [" << box.x << ", " << box.y << ", " << box.width << ", " << box.height << "]";
                    
                    if (formatInfo.fps > 0) {
                        double current_video_time_s = static_cast<double>(total_frames_read) / formatInfo.fps;
                        int total_seconds_int = static_cast<int>(current_video_time_s);
                        int minutes = total_seconds_int / 60;
                        int seconds = total_seconds_int % 60;
                        int milliseconds = static_cast<int>(std::fmod(current_video_time_s, 1.0) * 1000);
                        
                        std::cout << " (Video Time: " 
                                  << std::setfill('0') << std::setw(2) << minutes << ":"
                                  << std::setfill('0') << std::setw(2) << seconds << "."
                                  << std::setfill('0') << std::setw(3) << milliseconds << ")";
                    }
                    std::cout << std::endl;
                }
            }
            stage_end_time = std::chrono::high_resolution_clock::now();
            total_post_processing_time_us += std::chrono::duration_cast<std::chrono::microseconds>(stage_end_time - stage_start_time);
            
            auto single_pipeline_end_time = std::chrono::high_resolution_clock::now();
            cumulative_yolo_pipeline_time_us += std::chrono::duration_cast<std::chrono::microseconds>(single_pipeline_end_time - single_pipeline_start_time);
        } 
    } 

    auto overall_program_end_time = std::chrono::high_resolution_clock::now();
    auto total_program_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(overall_program_end_time - overall_program_start_time);
    double total_program_duration_s = total_program_duration_ms.count() / 1000.0;
    
    double effective_processing_rate_fps = 0.0;
    if (total_program_duration_s > 0 && yolo_processed_frame_count > 0) {
        effective_processing_rate_fps = static_cast<double>(yolo_processed_frame_count) / total_program_duration_s;
    }
    
    std::cout << "\n--- Processing Summary ---" << std::endl;
    std::cout << "Total frames read from video: " << total_frames_read << std::endl;
    std::cout << "Total frames processed by YOLO: " << yolo_processed_frame_count << std::endl;
    std::cout << "Total program run time: " << std::fixed << std::setprecision(2) << total_program_duration_s << " seconds" << std::endl;
    if (yolo_processed_frame_count > 0) {
         std::cout << "Effective rate of frames processed by YOLO: " << std::fixed << std::setprecision(2) << effective_processing_rate_fps << " FPS (approx. 1 frame per second target)" << std::endl;
    }

    std::cout << "\n--- OpenCV Performance Metrics (Avg. per frame) ---" << std::endl;
    if (total_frames_read > 0) {
        double avg_nextframe_ms = total_videoreader_nextframe_time_us.count() / (double)total_frames_read / 1000.0;
        std::cout << "  - Avg. VideoReader nextFrame() time: " 
                  << std::fixed << std::setprecision(3) << avg_nextframe_ms << " ms (Potential read FPS: " 
                  << std::setprecision(1) << (avg_nextframe_ms > 0.000001 ? 1000.0/avg_nextframe_ms : 0.0) << ")" << std::endl;
    }

    std::cout << "\n--- Detailed Time Consumption (Avg. per YOLO-Processed Frame) ---" << std::endl;
    if (yolo_processed_frame_count > 0) {
        double avg_total_yolo_pipeline_ms = cumulative_yolo_pipeline_time_us.count() / (double)yolo_processed_frame_count / 1000.0;
        std::cout << std::fixed << std::setprecision(3); 
        std::cout << "Avg. Total Pipeline Time for a YOLO-Processed Frame: " << avg_total_yolo_pipeline_ms << " ms" << std::endl;

        auto print_stage_stats = [&](const std::string& name, std::chrono::microseconds total_time_us, bool is_inference_stage = false) {
            if (yolo_processed_frame_count == 0) return;
            double avg_ms = total_time_us.count() / (double)yolo_processed_frame_count / 1000.0;
            double percentage = (avg_total_yolo_pipeline_ms > 0.000001) ? (avg_ms / avg_total_yolo_pipeline_ms * 100.0) : 0.0;
            std::cout << "  - Avg. " << std::setw(37) << std::left << name + ":" 
                      << std::setw(9) << std::right << avg_ms << " ms ("
                      << std::setw(6) << std::right << std::setprecision(2) << percentage << "%)";
            if (is_inference_stage && avg_ms > 0.000001) {
                std::cout << " [Potential Pure YOLO FPS: " << std::fixed << std::setprecision(1) << 1000.0 / avg_ms << "]";
            }
            std::cout << std::endl;
        };
        
        std::chrono::microseconds total_opencv_pre_yolo_processing_us = 
            total_gpu_conversion_time_us + 
            total_download_to_cpu_time_us + 
            total_blob_from_cpumat_time_us +
            total_set_input_time_us;

        print_stage_stats("OpenCV: GPU Color Conversion", total_gpu_conversion_time_us);
        print_stage_stats("OpenCV: GPU->CPU Download", total_download_to_cpu_time_us); 
        print_stage_stats("OpenCV: Blob Creation (from CPU Mat)", total_blob_from_cpumat_time_us); 
        print_stage_stats("OpenCV: DNN SetInput (CPU->GPU for blob)", total_set_input_time_us);
        std::cout << "    --------------------------------------------------------------------" << std::endl;
        print_stage_stats("Total OpenCV Pre-YOLO Processing", total_opencv_pre_yolo_processing_us);
        std::cout << "    --------------------------------------------------------------------" << std::endl;
        print_stage_stats("YOLO: DNN Forward (Inference)", total_forward_time_us, true); 
        print_stage_stats("YOLO: Post-processing (CPU)", total_post_processing_time_us);
    } else {
        std::cout << "No frames were processed by YOLO for detailed timing." << std::endl;
    }

    return 0;
}
