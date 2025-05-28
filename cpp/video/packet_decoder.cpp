#include "packet_decoder.h"
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

PacketDecoder::PacketDecoder(std::string video_file_name) :
    video_file_name(video_file_name),
    useHW(false),
    parser(nullptr),
    ctx(nullptr),
    codec(nullptr),
    fmtCtx(nullptr),
    hwCtxRef(nullptr),
    hw_device_ctx(nullptr),
    swsCtx(nullptr) {
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
    if (avformat_open_input(&fmtCtx, video_file_name.c_str(), nullptr, nullptr) != 0) {
        std::cerr << "Error: Could not open video file " << video_file_name << std::endl;
        return false;
    }
    if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
        std::cerr << "Error: Could not find stream information." << std::endl;
        avformat_close_input(&fmtCtx);
        return false;
    }

    int vidIdx = -1;

    for (unsigned i = 0; i < fmtCtx->nb_streams; i++) {
        if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            vidIdx = i;
            codec = avcodec_find_decoder(fmtCtx->streams[i]->codecpar->codec_id);
            if (!codec) {
                std::cerr << "Error: Unsupported codec!" << std::endl;
                avformat_close_input(&fmtCtx);
                return false;
            }
            break;
        }
    }

    ctx = avcodec_alloc_context3(codec);
    if (!ctx) {
        std::cerr << "Error: Could not allocate codec context." << std::endl;
        avformat_close_input(&fmtCtx);
        return false;
    }
    if (avcodec_parameters_to_context(ctx, fmtCtx->streams[vidIdx]->codecpar) < 0) {
        std::cerr << "Error: Could not copy codec parameters to context." << std::endl;
        avcodec_free_context(&ctx);
        avformat_close_input(&fmtCtx);
        return false;
    }

    ctx->thread_count = std::max(1u, std::thread::hardware_concurrency() / 2);
    useHW = initHWDecoder(ctx, &hwCtxRef);

    if (avcodec_open2(ctx, codec, nullptr) < 0) {
        std::cerr << "Error: Could not open codec." << std::endl;
        if (hwCtxRef) av_buffer_unref(&hwCtxRef);
        avcodec_free_context(&ctx);
        avformat_close_input(&fmtCtx);
        return false;
    }

    return true;
}

static AVPixelFormat getHWFormatCallback(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
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

int PacketDecoder::initHWDecoder(AVCodecContext *codecContext, AVBufferRef **hwDeviceCtx) {
    if (*hwDeviceCtx != nullptr) {
        av_buffer_unref(hwDeviceCtx);
        *hwDeviceCtx = nullptr;
    }
    const char *hwAccel[] = {"cuda", "vaapi", "vdpau", "qsv", "d3d11va", "videotoolbox", nullptr};
    for (int i = 0; hwAccel[i]; i++) {
        enum AVHWDeviceType type = av_hwdevice_find_type_by_name(hwAccel[i]);
        if (type == AV_HWDEVICE_TYPE_NONE) continue;
        int ret = av_hwdevice_ctx_create(hwDeviceCtx, type, 0, nullptr, 0);
        if (ret >= 0) {
            if (codecContext->hw_device_ctx) av_buffer_unref(&codecContext->hw_device_ctx);
            codecContext->hw_device_ctx = av_buffer_ref(*hwDeviceCtx);
            if (!codecContext->hw_device_ctx) {
                std::cerr << "Failed to create a new reference for hw_device_ctx for " << hwAccel[i] << std::endl;
                av_buffer_unref(hwDeviceCtx);
                *hwDeviceCtx = nullptr;
                continue;
            }
            codecContext->get_format = getHWFormatCallback;
            std::cout << "Hardware acceleration initialized successfully: " << hwAccel[i] << std::endl;
            return 1;
        }
    }
    std::cout << "No suitable hardware acceleration method was successfully initialized." << std::endl;
    return 0;
}

void PacketDecoder::decode(const std::vector<AVPacket *> &pkts, int interval) {
    AVFrame *frame = av_frame_alloc();
    AVFrame *swFrame = nullptr;
    // struct SwsContext *swsCtx = nullptr;
    if (!frame) {
        std::cerr << "Error: Could not allocate frame or packet." << std::endl;
        av_frame_free(&frame);
        av_frame_free(&swFrame);
        if (hwCtxRef) av_buffer_unref(&hwCtxRef);
        avcodec_close(ctx);
        avcodec_free_context(&ctx);
        avformat_close_input(&fmtCtx);
        return;
    }

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
}

std::vector<cv::Mat> PacketDecoder::getDecodedFrames() const {
    return decoded_frames;
}