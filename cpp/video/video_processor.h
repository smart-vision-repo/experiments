#ifndef VIDEO_PROCESSOR_H
#define VIDEO_PROCESSOR_H

#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

class VideoProcessor {
public:
    int process(const std::string& video_file_name, int interval);
};

#endif // VIDEO_PROCESSOR_H