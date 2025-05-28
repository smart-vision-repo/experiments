
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <memory>
#include <map>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <thread>
#include <algorithm>
#include <numeric>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/hwcontext.h>
#include <libavutil/error.h>
#include <libswscale/swscale.h>
}
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <video_file> <interval>" << std::endl;
        return -1;
    }

    int interval = 0;
    try {
        interval = std::stoi(argv[2]);
        if (interval <= 0) {
            std::cout << "Interval must be a positive integer." << std::endl;
            return -1;
        }
    } catch (const std::invalid_argument &ia) {
        std::cout << "Invalid interval argument: " << argv[2] << std::endl;
        return -1;
    } catch (const std::out_of_range &oor) {
        std::cout << "Interval argument out of range: " << argv[2] << std::endl;
        return -1;
    }

    std::string video_file_name = argv[1];
    if (!std::filesystem::exists(video_file_name)) {
        std::cout << "Video file does not exist: " << video_file_name << std::endl;
        return -1;
    }

    return process(video_file_name, interval);
}