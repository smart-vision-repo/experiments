#ifndef PACKET_DECODER_H
#define PACKET_DECODER_H

#include <vector>
#include <opencv2/core.hpp>

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
class PacketDecoder {
public:
    PacketDecoder(AVCodecID codec_id);
    ~PacketDecoder();

    void decode(const std::vector<AVPacket*>& pkts, int interval);
    std::vector<cv::Mat> getDecodedFrames() const;

private:
    bool initialize();
    int initializeHardwareDecoder(AVCodecContext *codecContext, AVBufferRef **hwDeviceCtx) ;
    AVPixelFormat getHWFormatCallback(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) ;
    AVCodecID codec_id;
    const AVCodec* codec;
    AVCodecContext* ctx;
    AVFormatContext *fmtCtx;
    AVCodecParserContext* parser;
    AVBufferRef* hw_device_ctx;
    SwsContext* swsCtx;
    std::vector<cv::Mat> decoded_frames;
};

#endif // PACKET_DECODER_H
