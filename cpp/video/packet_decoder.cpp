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
      sws_ctx_(nullptr) {}

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

bool PacketDecoder::decode(const std::vector<AVPacket*>& packets) {
    decoded_frames_.clear();

    for (AVPacket* pkt : packets) {
        if (avcodec_send_packet(ctx_, pkt) < 0) continue;

        while (true) {
            int ret = avcodec_receive_frame(ctx_, frame_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            else if (ret < 0)
                return false;

            AVFrame* frameToProcess = frame_;
            if (prefer_hw_ && frame_->format == ctx_->pix_fmt) {
                if (av_hwframe_transfer_data(sw_frame_, frame_, 0) < 0) {
                    av_frame_unref(frame_);
                    continue;
                }
                sw_frame_->width = frame_->width;
                sw_frame_->height = frame_->height;
                frameToProcess = sw_frame_;
            }

            if (!sws_ctx_) {
                sws_ctx_ = sws_getContext(
                    frameToProcess->width, frameToProcess->height, (AVPixelFormat)frameToProcess->format,
                    frameToProcess->width, frameToProcess->height, AV_PIX_FMT_BGR24,
                    SWS_BICUBIC, nullptr, nullptr, nullptr);
            }

            cv::Mat img(frameToProcess->height, frameToProcess->width, CV_8UC3);
            uint8_t* dst_data[1] = { img.data };
            int dst_linesize[1] = { (int)img.step[0] };
            sws_scale(sws_ctx_, frameToProcess->data, frameToProcess->linesize,
                      0, frameToProcess->height, dst_data, dst_linesize);
            decoded_frames_.push_back(img);

            av_frame_unref(frame_);
            if (frameToProcess == sw_frame_) av_frame_unref(sw_frame_);
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