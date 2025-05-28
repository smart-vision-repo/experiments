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
        uint8_t *data = pkt->data;
        int size = pkt->size;
        
        while (size > 0) {
            int ret = av_parser_parse2(
                parser, ctx,
                &parsed_pkt->data, &parsed_pkt->size,
                data, size,
                pkt->pts, pkt->dts, pkt->pos);

            if (ret < 0) {
                std::cerr << "Parser error." << std::endl;
                break;
            }

            data += ret;
            size -= ret;

            std::cerr << "pkt size." << parsed_pkt->size << std::endl;
            if (parsed_pkt->size > 0) {
                if (avcodec_send_packet(ctx, parsed_pkt) < 0) {
                    std::cerr << "Failed to send packet." << std::endl;
                    continue;
                }

                while (true) {
                    int response = avcodec_receive_frame(ctx, frame);

                    if (response == AVERROR(EAGAIN)) {
                        // 数据不足，尝试下一 packet
                        break;
                    } else if (response == AVERROR_EOF) {
                        // 解码完成
                        break;
                    } else if (response < 0) {
                        std::cerr << "Error while receiving frame: " << response << std::endl;
                        break;
                    }

                    // 如果能解出帧，处理它
                    std::cout << "Decoded one frame: " << frame->width << "x" << frame->height << std::endl;

                    // 注意这里只是占位，应该转换为 cv::Mat
                    decoded_frames.push_back(cv::Mat());

                    av_frame_unref(frame); // 准备下一帧
                }
            }
        }
    }

    avcodec_send_packet(ctx, nullptr); // Flush
    while (true) {
        int ret = avcodec_receive_frame(ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            break;
        else if (ret < 0)
            break;
        decoded_frames.push_back(cv::Mat()); // Placeholder
        av_frame_unref(frame);
    }

    av_frame_free(&frame);
    av_packet_free(&parsed_pkt);
}

std::vector<cv::Mat> PacketDecoder::getDecodedFrames() const {
    return decoded_frames;
}