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

static inline std::string ffmpeg_error_string(int errnum) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errnum, errbuf, sizeof(errbuf));
    return std::string(errbuf);
}

class PacketDecoder {
public:
    PacketDecoder(AVCodecID codec_id) :
        codec_id(codec_id) {
    }
    ~PacketDecoder() {
        cleanup();
    }

    bool initialize() {
        codec = avcodec_find_decoder_by_name("h264_cuvid");
        if (codec) {
            ctx = avcodec_alloc_context3(codec);
            if (av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) == 0) {
                ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
                ctx->get_format = getHWFormatCallback;
                use_hw = true;
            } else {
                avcodec_free_context(&ctx);
                codec = nullptr;
            }
        }
        if (!codec) {
            codec = avcodec_find_decoder(codec_id);
            if (!codec) return false;
            ctx = avcodec_alloc_context3(codec);
            use_hw = false;
        }
        return avcodec_open2(ctx, codec, nullptr) >= 0;
    }

    bool decode(std::vector<AVPacket *> *pkts) {
        frame = av_frame_alloc();
        sw_frame = av_frame_alloc();
        if (!frame || !sw_frame) return false;

        for (AVPacket *pkt : *pkts) {
            if (avcodec_send_packet(ctx, pkt) < 0) continue;
            while (true) {
                int ret = avcodec_receive_frame(ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                    break;
                else if (ret < 0)
                    break;

                AVFrame *f = frame;
                if (use_hw && frame->format == ctx->pix_fmt) {
                    if (av_hwframe_transfer_data(sw_frame, frame, 0) == 0) {
                        f = sw_frame;
                    } else {
                        std::cerr << "Failed to transfer HW frame to SW frame." << std::endl;
                        continue;
                    }
                }
                std::cout << "Decoded frame: " << f->width << "x" << f->height << std::endl;
                av_frame_unref(frame);
                if (use_hw) av_frame_unref(sw_frame);
            }
        }

        avcodec_send_packet(ctx, nullptr);
        while (true) {
            int ret = avcodec_receive_frame(ctx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            av_frame_unref(frame);
            if (use_hw) av_frame_unref(sw_frame);
        }
        return true;
    }

private:
    void cleanup() {
        if (ctx) avcodec_free_context(&ctx);
        if (frame) av_frame_free(&frame);
        if (sw_frame) av_frame_free(&sw_frame);
        if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);
    }

    AVCodecID codec_id;
    AVCodec *codec = nullptr;
    AVCodecContext *ctx = nullptr;
    AVBufferRef *hw_device_ctx = nullptr;
    AVFrame *frame = nullptr;
    AVFrame *sw_frame = nullptr;
    bool use_hw = false;
};
