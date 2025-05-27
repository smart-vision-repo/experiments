#include <iostream>
#include <string>
#include <stdexcept> // For std::stoi exception handling

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

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

    AVFormatContext *formatContext = nullptr;
    if (avformat_open_input(&formatContext, argv[1], nullptr, nullptr) < 0) {
        std::cout << "Could not open video file: " << argv[1] << std::endl;
        return -1;
    }

    if (avformat_find_stream_info(formatContext, nullptr) < 0) {
        std::cout << "Could not get stream info" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    int videoStream = -1;
    for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStream = i;
            break;
        }
    }

    if (videoStream == -1) {
        std::cout << "No video stream found" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    AVPacket *packet = av_packet_alloc();
    if (!packet) {
        std::cout << "Could not allocate AVPacket" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    int frame_idx = 0;
    int gop_idx = 0;
    int frame_idx_in_gop = 0;
    int hits = 0;
    int pool = 0;

    int total_hits = 0;
    int decoded_frames = 0;
    int skipped_frames = 0;

    while (av_read_frame(formatContext, packet) >= 0) {
        if (packet->stream_index == videoStream) {
            frame_idx++;
            if (frame_idx > 1 && (frame_idx - 1) % interval == 0) {
                hits++;
            }
            bool isKeyFrame = (packet->flags & AV_PKT_FLAG_KEY);
            if (isKeyFrame) {
                int last_frame_in_gop = 0;
                if (hits > 0) {
                    skipped_frames += pool;
                    last_frame_in_gop = hits * interval - pool;
                    decoded_frames += last_frame_in_gop;
                    total_hits += hits;
                    pool = frame_idx_in_gop - last_frame_in_gop;
                } else {
                    pool += frame_idx_in_gop;
                }
                /*
                std::cout << gop_idx << ","
                          << frame_idx_in_gop << ","
                          << hits << ","
                          << frame_idx - 1 << ","
                          << last_frame_in_gop << ","
                          << pool << std::endl;
                */
                frame_idx_in_gop = 0;
                hits = 0;
                gop_idx++;
            }
            frame_idx_in_gop++;
            av_packet_unref(packet);
        }
    }

    int last_frame_in_gop = 0;
    if (hits > 0) {
        last_frame_in_gop = hits * interval - pool;
        if (last_frame_in_gop > 0) {
            decoded_frames += last_frame_in_gop;
        }
        total_hits += hits;
        pool = frame_idx_in_gop - last_frame_in_gop;
    } else {
        pool += frame_idx_in_gop;
    }
    skipped_frames += pool;
    /*
    std::cout << gop_idx << ","
              << frame_idx_in_gop << ","
              << hits << ","
              << frame_idx - 1 << ","
              << last_frame_in_gop << ","
              << pool << std::endl;
    */

    av_packet_free(&packet);
    avformat_close_input(&formatContext);

    std::cout << "-------------------" << std::endl;

    float percentage = (frame_idx > 0) ? (decoded_frames * 100.0f / frame_idx) : 0.0f;
    std::cout << "total gop: " << gop_idx << std::endl
              << "total frames: " << frame_idx << std::endl
              << "decoded frames: " << decoded_frames << std::endl
              << "skipped frames: " << skipped_frames << std::endl
              << "computed frames: " << decoded_frames + skipped_frames << std::endl
              << "discrepancies: " << frame_idx - decoded_frames - skipped_frames << std::endl
              << "percentage: " << percentage << "%" << std::endl
              << "extracted frames: " << total_hits + pool << std::endl;
    return 0;
}