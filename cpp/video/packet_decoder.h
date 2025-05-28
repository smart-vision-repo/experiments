#ifndef PACKET_DECODER_H
#define PACKET_DECODER_H

#include <vector>
#include <opencv2/core.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
}

class PacketDecoder {
public:
    PacketDecoder(AVCodecID codec_id, bool prefer_hw = true);
    ~PacketDecoder();

    bool initialize();
    bool decode(const std::vector<AVPacket*>& packets, int interval);
    std::vector<cv::Mat> getDecodedFrames() const;

private:
    AVCodecID codec_id_;
    bool prefer_hw_;
    AVCodecContext* ctx_;
    AVBufferRef* hw_device_ctx_;
    AVFrame* frame_;
    AVFrame* sw_frame_;
    struct SwsContext* sws_ctx_;
    std::vector<cv::Mat> decoded_frames_;

    static AVPixelFormat getHWFormatCallback(AVCodecContext* ctx, const AVPixelFormat* pix_fmts);
    void releaseResources();
};

#endif // PACKET_DECODER_H
