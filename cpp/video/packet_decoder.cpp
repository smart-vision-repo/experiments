#include "packet_decoder.h"
#include <iostream>

PacketDecoder::PacketDecoder(AVCodecID codec_id) :
    codec_id(codec_id), ctx(nullptr), parser(nullptr), codec(nullptr), swsCtx(nullptr), hw_device_ctx(nullptr) {
    if (!initialize()) {
        throw std::runtime_error("Failed to initialize PacketDecoder.");
    }
}

PacketDecoder::~PacketDecoder() {
    if (ctx) {
        avcodec_free_context(&ctx);
    }
    if (parser) {
        av_parser_close(parser);
    }
    if (swsCtx) {
        sws_freeContext(swsCtx);
    }
    if (hw_device_ctx) {
        av_buffer_unref(&hw_device_ctx);
    }
}

bool PacketDecoder::initialize() {
    codec = avcodec_find_decoder(codec_id);
    if (!codec) {
        std::cerr << "Codec not found." << std::endl;
        return false;
    }

    parser = av_parser_init(codec_id);
    if (!parser) {
        std::cerr << "Parser not found for codec." << std::endl;
        return false;
    }

    ctx = avcodec_alloc_context3(codec);
    if (!ctx) {
        std::cerr << "Could not allocate codec context." << std::endl;
        return false;
    }

    if (avcodec_open2(ctx, codec, nullptr) < 0) {
        std::cerr << "Could not open codec." << std::endl;
        return false;
    }

    return true;
}

void PacketDecoder::decode(const std::vector<AVPacket *> &pkts, int interval) {
    AVFrame *frame = av_frame_alloc();
    AVPacket *parsed_pkt = av_packet_alloc();

    for (AVPacket *pkt : pkts) {
        avcodec_send_packet(ctx, pkt); // Flush
        while (true) {
            int ret = avcodec_receive_frame(ctx, frame);
            std::cout << ret << std::endl;
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            else if (ret < 0)
                break;
            decoded_frames.push_back(cv::Mat()); // Placeholder
            av_frame_unref(frame);
        }
    }

    av_frame_free(&frame);
    av_packet_free(&parsed_pkt);
}

std::vector<cv::Mat> PacketDecoder::getDecodedFrames() const {
    return decoded_frames;
}