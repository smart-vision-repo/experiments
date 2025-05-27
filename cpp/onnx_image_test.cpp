#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>   // For std::fixed and std::setprecision
#include <algorithm> // For std::max_element

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// YOLO Model Constants
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const int DOG_CLASS_ID = 16; // COCO class ID for "dog" (0-indexed)

std::vector<std::string> load_class_names(const std::string &path) {
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

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_file_path>" << std::endl;
        std::cerr << "Example: " << argv[0] << " test_dog_image.jpg" << std::endl;
        return -1;
    }

    std::string image_file = argv[1];

    std::string model_path = "yolov8n.onnx";
    std::string class_names_path = "coco.names";

    std::vector<std::string> class_names = load_class_names(class_names_path);
    if (class_names.empty() || class_names.size() <= DOG_CLASS_ID) {
        std::cerr << "Error: Class names not loaded correctly or 'dog' class ID is out of bounds." << std::endl;
        if (!class_names.empty()) {
            std::cerr << "Loaded " << class_names.size() << " classes, expected at least " << DOG_CLASS_ID + 1 << std::endl;
        }
        return -1;
    }
    int num_classes = class_names.size();
    std::cout << "Loaded " << num_classes << " class names. Target class 'dog' is ID " << DOG_CLASS_ID << "." << std::endl;

    cv::Mat image = cv::imread(image_file);
    if (image.empty()) {
        std::cerr << "Error: Could not read image: " << image_file << std::endl;
        return -1;
    }
    std::cout << "Successfully loaded image: " << image_file << " (" << image.cols << "x" << image.rows << ")" << std::endl;

    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(model_path);
        std::cout << "Successfully loaded ONNX model: " << model_path << std::endl;
    } catch (const cv::Exception &e) {
        std::cerr << "Error loading ONNX model: " << e.what() << std::endl;
        return -1;
    }

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    std::cout << "Using CUDA backend for DNN inference." << std::endl;

    cv::Mat blob;
    // Preprocessing: Resize, BGR->RGB, Normalize, HWC->NCHW
    // swapRB=true handles BGR to RGB
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false, CV_32F);
    std::cout << "Blob created with shape: " << blob.size << std::endl; // Should be [1, 3, 640, 640]

    net.setInput(blob);

    std::vector<cv::Mat> outs;
    try {
        net.forward(outs, net.getUnconnectedOutLayersNames());
        std::cout << "Model forward pass completed. Output layers: " << outs.size() << std::endl;
    } catch (const cv::Exception &e) {
        std::cerr << "Error during model forward pass: " << e.what() << std::endl;
        return -1;
    }

    if (outs.empty() || outs[0].empty()) {
        std::cerr << "Error: Model output is empty." << std::endl;
        return -1;
    }

    // YOLOv8 output is typically [batch_size, num_classes + 4, num_proposals]
    // e.g., [1, 84, 8400] for COCO (80 classes + 4 bbox coords)
    // We need to transpose it to [batch_size, num_proposals, num_classes + 4] for easier iteration
    cv::Mat output_layer = outs[0];
    std::cout << "Raw output layer shape: " << output_layer.size << std::endl;

    if (output_layer.dims < 3 || output_layer.size[0] != 1 || output_layer.size[1] != (4 + num_classes) || output_layer.size[2] == 0) {
        std::cerr << "Error: Unexpected output layer shape from YOLO model. Expected: [1][" << (4 + num_classes) << "][N > 0], Got: [";
        for (int i = 0; i < output_layer.dims; ++i) std::cerr << output_layer.size[i] << (i == output_layer.dims - 1 ? "" : ", ");
        std::cerr << "]" << std::endl;
        return -1;
    }

    // Transpose to get [num_proposals, num_classes + 4]
    // outs[0] is Mat of (1, 84, 8400)
    // temp_mat reshapes to (84, 8400)
    // detections_transposed then becomes (8400, 84)
    cv::Mat temp_mat = output_layer.reshape(1, output_layer.size[1]);
    cv::Mat detections_transposed;
    cv::transpose(temp_mat, detections_transposed);
    std::cout << "Transposed detections matrix shape: " << detections_transposed.size << std::endl;

    int num_proposals = detections_transposed.rows;
    bool dog_signal_found = false;
    float max_dog_score = 0.0f;
    int dog_proposal_idx = -1;

    std::cout << "\n--- Checking Raw Model Outputs for 'dog' (Class ID " << DOG_CLASS_ID << ") ---" << std::endl;

    for (int i = 0; i < num_proposals; ++i) {
        cv::Mat proposal_row = detections_transposed.row(i); // Shape: 1 x (4 + num_classes)
        // The first 4 elements are cx, cy, w, h.
        // The next num_classes elements are the class scores.
        const float *scores_ptr = proposal_row.ptr<float>(0, 4); // Pointer to the first class score

        float current_dog_score = scores_ptr[DOG_CLASS_ID];

        if (current_dog_score > 0.01f) { // Print any proposal with even a slight "dog" score
            std::cout << "Proposal " << i << ": 'dog' (ID " << DOG_CLASS_ID << ") raw score = "
                      << std::fixed << std::setprecision(4) << current_dog_score << std::endl;
            dog_signal_found = true;
            if (current_dog_score > max_dog_score) {
                max_dog_score = current_dog_score;
                dog_proposal_idx = i;
            }
        }

        // Optional: Print top class for this proposal if its score is high
        if (i < 5 || current_dog_score > 0.01f) { // Limit printing for brevity, or if it's a dog
            float max_score_for_proposal = 0.f;
            int max_class_id_for_proposal = -1;
            for (int j = 0; j < num_classes; ++j) {
                if (scores_ptr[j] > max_score_for_proposal) {
                    max_score_for_proposal = scores_ptr[j];
                    max_class_id_for_proposal = j;
                }
            }
            if (max_score_for_proposal > 0.1) { // Print if any class has a decent score
                std::cout << "  Proposal " << i << ": Top class is '"
                          << (max_class_id_for_proposal < num_classes ? class_names[max_class_id_for_proposal] : "Unknown")
                          << "' (ID " << max_class_id_for_proposal << ") with score "
                          << std::fixed << std::setprecision(4) << max_score_for_proposal << std::endl;
            }
        }
    }

    if (dog_signal_found) {
        std::cout << "\nSUCCESS: 'dog' signal found in raw model output! Max score: " << max_dog_score
                  << " at proposal index " << dog_proposal_idx << std::endl;
    } else {
        std::cout << "\nFAILURE: No significant 'dog' signal found in raw model output (checked "
                  << num_proposals << " proposals with threshold > 0.01)." << std::endl;
    }
    std::cout << "--- End Raw Output Check ---" << std::endl;

    return 0;
}
