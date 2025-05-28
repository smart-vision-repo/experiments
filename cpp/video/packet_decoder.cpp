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

    AVPixelFormat getHWFormatCallback(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
        if (!ctx || !ctx->hw_device_ctx || !ctx->hw_device_ctx->data) {
            std::cerr << "getHWFormatCallback: ERROR - Missing hardware device context in AVCodecContext." << std::endl;
            return AV_PIX_FMT_NONE;
        }
        AVHWDeviceContext *hw_device_ctx_data = (AVHWDeviceContext *)ctx->hw_device_ctx->data;
        const enum AVHWDeviceType active_device_type = hw_device_ctx_data->type;
        std::cout << "getHWFormatCallback: Active HW device type is '"
                  << av_hwdevice_get_type_name(active_device_type)
                  << "'. Searching for compatible pixel format..." << std::endl;

        for (const enum AVPixelFormat *p_fmt = pix_fmts; *p_fmt != AV_PIX_FMT_NONE; ++p_fmt) {
            std::cout << "  - Considering format: " << av_get_pix_fmt_name(*p_fmt) << std::endl;
            bool format_is_compatible = false;
            switch (active_device_type) {
            case AV_HWDEVICE_TYPE_CUDA:
                if (*p_fmt == AV_PIX_FMT_CUDA) format_is_compatible = true;
                break;
            case AV_HWDEVICE_TYPE_VAAPI:
                if (*p_fmt == AV_PIX_FMT_VAAPI) format_is_compatible = true;
                break;
            case AV_HWDEVICE_TYPE_VDPAU:
                if (*p_fmt == AV_PIX_FMT_VDPAU) format_is_compatible = true;
                break;
            case AV_HWDEVICE_TYPE_QSV:
                if (*p_fmt == AV_PIX_FMT_QSV) format_is_compatible = true;
                break;
            case AV_HWDEVICE_TYPE_D3D11VA:
                if (*p_fmt == AV_PIX_FMT_D3D11) format_is_compatible = true;
                break;
            case AV_HWDEVICE_TYPE_VIDEOTOOLBOX:
                if (*p_fmt == AV_PIX_FMT_VIDEOTOOLBOX) format_is_compatible = true;
                break;
            default:
                std::cerr << "getHWFormatCallback: Unhandled or unknown active_device_type: "
                          << av_hwdevice_get_type_name(active_device_type) << std::endl;
                break;
            }
            if (format_is_compatible) {
                std::cout << "Hardware pixel format selected: " << av_get_pix_fmt_name(*p_fmt)
                          << " (Compatible with " << av_hwdevice_get_type_name(active_device_type) << ")" << std::endl;
                return *p_fmt;
            }
        }
        std::cerr << "getHWFormatCallback: Failed to find a suitable HW pixel format for device type: "
                  << av_hwdevice_get_type_name(active_device_type) << std::endl;
        std::cerr << "  Available formats in pix_fmts were: ";
        for (const enum AVPixelFormat *p_fmt = pix_fmts; *p_fmt != AV_PIX_FMT_NONE; ++p_fmt) {
            std::cerr << av_get_pix_fmt_name(*p_fmt) << " ";
        }
        std::cerr << std::endl;
        return AV_PIX_FMT_NONE;
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
