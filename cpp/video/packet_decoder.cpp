// packet_decoder.cpp

#include "packet_decoder.h"
#include <iostream>
#include <opencv2/imgproc.hpp>

extern "C" {
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

PacketDecoder::PacketDecoder(AVCodecID codec_id, bool prefer_hw)
    : codec_id_(codec_id), prefer_hw_(prefer_hw), ctx_(nullptr),
      hw_device_ctx_(nullptr), frame_(nullptr), sw_frame_(nullptr),
      sws_ctx_(nullptr) {
        initialize();
      }

PacketDecoder::~PacketDecoder() {
    releaseResources();
}

bool PacketDecoder::initialize() {
    const AVCodec* codec = avcodec_find_decoder(codec_id_);
    if (!codec) {
        std::cerr << "Codec not found." << std::endl;
        return false;
    }

    ctx_ = avcodec_alloc_context3(codec);
    if (!ctx_) {
        std::cerr << "Failed to allocate codec context." << std::endl;
        return false;
    }

    if (prefer_hw_) {
        if (av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) >= 0) {
            ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
            ctx_->get_format = [](AVCodecContext* ctx, const AVPixelFormat* pix_fmts) {
                for (const AVPixelFormat* p = pix_fmts; *p != -1; p++) {
                    if (*p == AV_PIX_FMT_CUDA) {
                        return *p;
                    }
                }
                return pix_fmts[0];
            };
        } else {
            std::cerr << "HW decoding not available, falling back to SW." << std::endl;
        }
    }

    if (avcodec_open2(ctx_, codec, nullptr) < 0) {
        std::cerr << "Failed to open codec." << std::endl;
        return false;
    }

    frame_ = av_frame_alloc();
    sw_frame_ = av_frame_alloc();
    return frame_ && sw_frame_;
}


bool PacketDecoder::decode(const std::vector<AVPacket*>& packets, int interval) {
    decoded_frames_.clear();
    // Annotate the processing steps within the decode method

    // Clear any previously decoded frames to start fresh
    decoded_frames_.clear();

    // Calculate the indices of frames to be processed based on the interval
    int frame_count = packets.size();
    std::vector<int> frame_indices;
    while(frame_count > 0) {
        frame_indices.push_back(frame_count);
        frame_count -= interval;
    }
    frame_indices.push_back(0);
    std::sort(frame_indices.begin(), frame_indices.end());

    // Initialize index for tracking packet processing
    int index = 0;

    // Iterate over each packet to decode
    for (AVPacket* pkt : packets) {
        // Send the packet to the decoder
        if (avcodec_send_packet(ctx_, pkt) < 0) continue;

        // Receive frames from the decoder
        while (true) {
            int ret = avcodec_receive_frame(ctx_, frame_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;  // No more frames available or end of file
            else if (ret < 0)
                return false;  // Error in receiving frame

            // Determine which frame to process
            AVFrame* frameToProcess = frame_;
            std::cout << "package decoded:" << frame_->pkt_size << std::endl;
            // Unreference the frames to free up resources
            av_frame_unref(frame_);
            av_frame_unref(sw_frame_);
        }
    }

    avcodec_send_packet(ctx_, nullptr);
    while (true) {
        int ret = avcodec_receive_frame(ctx_, frame_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        av_frame_unref(frame_);
    }

    return true;
}

std::vector<cv::Mat> PacketDecoder::getDecodedFrames() const {
    return decoded_frames_;
}

void PacketDecoder::releaseResources() {
    if (ctx_) {
        avcodec_free_context(&ctx_);
    }
    if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
    }
    if (frame_) {
        av_frame_free(&frame_);
    }
    if (sw_frame_) {
        av_frame_free(&sw_frame_);
    }
    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
    }
}

AVPixelFormat PacketDecoder::getHWFormatCallback(AVCodecContext* ctx, const AVPixelFormat* pix_fmts) {
    for (const AVPixelFormat* p = pix_fmts; *p != -1; p++) {
        if (*p == AV_PIX_FMT_CUDA) {
            return *p;
        }
    }
    return pix_fmts[0];
}