#ifndef VIDEO_PROCESSOR_H
#define VIDEO_PROCESSOR_H

#include <string>
#include <vector>
extern "C" {
#include <libavcodec/avcodec.h>
}

// 声明 VideoProcessor 类
class VideoProcessor {
public:
    int process(const std::string& video_file_name, int interval);

private:
    void add_av_packet_to_list(std::vector<AVPacket*>** packages, const AVPacket* packet);
    std::vector<AVPacket*> get_packets_for_decoding(std::vector<AVPacket*>* packages, int last_frame_index);
    void clear_av_packets(std::vector<AVPacket*>* packages);
};

#endif // VIDEO_PROCESSOR_H
