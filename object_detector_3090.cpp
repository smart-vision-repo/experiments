#define SAVE_IMAGE_COUNT 10
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

namespace fs = std::filesystem;

struct GOPSamplingInfo {
    int gopIndex = 0;
    int framesInGop = 0;
    int samplesToTake = 0;
    std::vector<int> frameIndicesToDecode;
    int lastSampledOrdinal = 0;
    int remainingFramesAfterD = 0;
    long long next_carry_over = 0;
};

void createOutputDirectory(const std::string& outputDir) {
    if (!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
        std::cout << "Created output directory: " << outputDir << std::endl;
    }
}

std::string generateTimestampFilename(const std::string& prefix, int gopIndex, int frameIndexInGop, const std::string& extension) {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_c);
    std::stringstream ss;
    ss << prefix << "_gop" << std::setw(3) << std::setfill('0') << gopIndex
       << "_frame" << std::setw(3) << std::setfill('0') << frameIndexInGop
       << "_" << std::put_time(&now_tm, "%Y%m%d_%H%M%S")
       << extension;
    return ss.str();
}

// MODIFIED getHWFormatCallback
AVPixelFormat getHWFormatCallback(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
    // Check if hw_device_ctx and its data are valid
    if (!ctx || !ctx->hw_device_ctx || !ctx->hw_device_ctx->data) {
        std::cerr << "getHWFormatCallback: ERROR - Missing hardware device context in AVCodecContext." << std::endl;
        return AV_PIX_FMT_NONE;
    }

    // Get the type of the initialized hardware device context
    // The actual device context data is pointed to by AVBufferRef->data,
    // and this needs to be cast to AVHWDeviceContext to get its 'type'.
    AVHWDeviceContext *hw_device_ctx_data = (AVHWDeviceContext*)ctx->hw_device_ctx->data;
    const enum AVHWDeviceType active_device_type = hw_device_ctx_data->type;

    std::cout << "getHWFormatCallback: Active HW device type is '"
              << av_hwdevice_get_type_name(active_device_type)
              << "'. Searching for compatible pixel format..." << std::endl;

    // Iterate through the pixel formats provided by FFmpeg
    for (const enum AVPixelFormat *p_fmt = pix_fmts; *p_fmt != AV_PIX_FMT_NONE; ++p_fmt) {
        std::cout << "  - Considering format: " << av_get_pix_fmt_name(*p_fmt) << std::endl;
        bool format_is_compatible = false;
        switch (active_device_type) {
            case AV_HWDEVICE_TYPE_CUDA:
                if (*p_fmt == AV_PIX_FMT_CUDA) format_is_compatible = true;
                break;
            case AV_HWDEVICE_TYPE_VAAPI:
                if (*p_fmt == AV_PIX_FMT_VAAPI) format_is_compatible = true;
                // Potentially other VAAPI related formats like AV_PIX_FMT_NV12 might be considered
                // if they represent VAAPI surfaces, but AV_PIX_FMT_VAAPI is the primary one.
                break;
            case AV_HWDEVICE_TYPE_VDPAU:
                if (*p_fmt == AV_PIX_FMT_VDPAU) format_is_compatible = true;
                break;
            case AV_HWDEVICE_TYPE_QSV:
                if (*p_fmt == AV_PIX_FMT_QSV) format_is_compatible = true;
                break;
            case AV_HWDEVICE_TYPE_D3D11VA:
                // Note: The pixel format for D3D11VA is AV_PIX_FMT_D3D11
                if (*p_fmt == AV_PIX_FMT_D3D11) format_is_compatible = true;
                break;
            case AV_HWDEVICE_TYPE_VIDEOTOOLBOX:
                if (*p_fmt == AV_PIX_FMT_VIDEOTOOLBOX) format_is_compatible = true;
                break;
            // Add other hardware types if you plan to support them
            default:
                std::cerr << "getHWFormatCallback: Unhandled or unknown active_device_type: "
                          << av_hwdevice_get_type_name(active_device_type) << std::endl;
                break;
        }

        if (format_is_compatible) {
            std::cout << "Hardware pixel format selected: " << av_get_pix_fmt_name(*p_fmt)
                      << " (Compatible with " << av_hwdevice_get_type_name(active_device_type) << ")" << std::endl;
            return *p_fmt; // Return the first compatible format found
        }
    }

    std::cerr << "getHWFormatCallback: Failed to find a suitable HW pixel format for device type: "
              << av_hwdevice_get_type_name(active_device_type) << std::endl;
    std::cerr << "  Available formats in pix_fmts were: ";
    for (const enum AVPixelFormat *p_fmt = pix_fmts; *p_fmt != AV_PIX_FMT_NONE; ++p_fmt) {
        std::cerr << av_get_pix_fmt_name(*p_fmt) << " ";
    }
    std::cerr << std::endl;

    return AV_PIX_FMT_NONE; // No compatible format found
}


int initializeHardwareDecoder(AVCodecContext* codecContext, AVBufferRef** hwDeviceCtx) {
    // Ensure hwDeviceCtx is initialized to NULL if it's going to be populated by av_hwdevice_ctx_create
    if (*hwDeviceCtx != nullptr) {
        av_buffer_unref(hwDeviceCtx); // Should not happen if used correctly, but for safety
        *hwDeviceCtx = nullptr;
    }

    const char* hwAccel[] = {"cuda", "vaapi", "vdpau", "qsv", "d3d11va", "videotoolbox", nullptr};
    for (int i = 0; hwAccel[i]; i++) {
        enum AVHWDeviceType type = av_hwdevice_find_type_by_name(hwAccel[i]);
        if (type == AV_HWDEVICE_TYPE_NONE) {
            // This is not an error, just means this accel method is not compiled in or found by name
            // std::cout << "HWAccel: " << hwAccel[i] << " not found by name." << std::endl;
            continue;
        }

        // Attempt to create the hardware device context.
        // av_hwdevice_ctx_create will allocate and fill *hwDeviceCtx.
        int ret = av_hwdevice_ctx_create(hwDeviceCtx, type, 0, nullptr, 0);
        if (ret >= 0) {
            // Successfully created a hardware device context.
            // Now, assign a reference to this context to the codec context.
            if (codecContext->hw_device_ctx) { // If codec context already has one, unref it first.
                 av_buffer_unref(&codecContext->hw_device_ctx);
            }
            codecContext->hw_device_ctx = av_buffer_ref(*hwDeviceCtx);
            if (!codecContext->hw_device_ctx) {
                std::cerr << "Failed to create a new reference for hw_device_ctx for " << hwAccel[i] << std::endl;
                av_buffer_unref(hwDeviceCtx); // Clean up the context created by av_hwdevice_ctx_create
                *hwDeviceCtx = nullptr; // Reset the pointer
                // Continue to try next hw accel method or return error? For now, let's try next.
                continue;
            }

            codecContext->get_format = getHWFormatCallback; // Set our callback
            std::cout << "Hardware acceleration initialized successfully: " << hwAccel[i] << std::endl;
            return 1; // Indicate success
        } else {
            // Failed to create context for this type, try next.
            // Log detailed error from FFmpeg.
            std::cout << "Failed to create HW device context for " << hwAccel[i] << ". Error: " << ffmpeg_error_string(ret) << std::endl;
            // *hwDeviceCtx might be partially initialized or invalid, ensure it's NULL for next iteration if av_hwdevice_ctx_create failed.
            // However, av_hwdevice_ctx_create on failure should leave *hwDeviceCtx untouched or NULL.
            // If it was modified, it would be an FFmpeg bug. For safety, one might consider av_buffer_unref(hwDeviceCtx) and *hwDeviceCtx = nullptr here too,
            // but it's generally not needed if av_hwdevice_ctx_create behaves as documented.
        }
    }
    std::cout << "No suitable hardware acceleration method was successfully initialized." << std::endl;
    return 0; // Indicate failure to initialize any hardware acceleration
}

GOPSamplingInfo calculate_sample_indices_for_gop(int gop_idx, int b_val, long long initial_carry, int interval, bool is_last_gop) {
    GOPSamplingInfo result;
    result.gopIndex = gop_idx;
    result.framesInGop = b_val;
    long long pool = initial_carry + b_val;
    int c_val = (interval>0) ? pool/interval : 0;
    int d_val = 0;
    if (c_val>0) {
        long long first = interval - initial_carry;
        if (first<=0) first=1; // Ensure first is at least 1 if interval <= initial_carry
        // Calculate d_val based on how many full intervals fit
        if (b_val >= first) { // Only if there's room for the first sample
             d_val = first + (long long)((c_val-1)>0?(c_val-1):0)*interval; //Ordinal of the last sample
             // Ensure d_val does not exceed b_val
             if (d_val > b_val) {
                 // This case means c_val was too high, recalculate based on actual frames
                 if (b_val < first) { // Not even the first sample fits
                     d_val = 0;
                 } else {
                     d_val = first + (long long)((b_val - first) / interval) * interval;
                 }
             }
        } else {
            d_val = 0; // Not enough frames for even the first adjusted sample
        }

        if (d_val < 1) d_val = 0; // Correct if d_val became non-positive
    }

    result.lastSampledOrdinal = d_val;
    // Calculate samplesToTake based on d_val and first
    if (d_val > 0 && d_val >= (interval - initial_carry)) {
         result.samplesToTake = ((d_val - (interval - initial_carry)) / interval) + 1;
    } else {
        result.samplesToTake = 0;
    }
    
    for (int i=0; i<result.samplesToTake; i++) {
        result.frameIndicesToDecode.push_back((interval-initial_carry) + i*interval);
    }
    result.remainingFramesAfterD = (d_val>0)? (b_val-d_val):b_val; // if d_val is 0, all frames are remaining
    result.next_carry_over = is_last_gop?0:(pool%interval);
    if (pool < interval && !is_last_gop) { // If pool itself is less than interval, it's all carry over
        result.next_carry_over = pool;
    }


    // Debug print
    // std::cout << "GOP: " << gop_idx << ", B: " << b_val << ", CarryIn: " << initial_carry
    //           << ", Interval: " << interval << ", Pool: " << pool << ", C: " << c_val
    //           << ", D: " << d_val << ", Samples: " << result.samplesToTake
    //           << ", CarryOut: " << result.next_carry_over << std::endl;
    // std::cout << "  Sample indices: ";
    // for(int idx : result.frameIndicesToDecode) std::cout << idx << " ";
    // std::cout << std::endl;

    return result;
}

void processAndDecodeVideo(const std::string& videoFile, int interval, const std::string& outputDir) {
    createOutputDirectory(outputDir);
    AVFormatContext* fmtCtx = nullptr;
    if (avformat_open_input(&fmtCtx, videoFile.c_str(), nullptr, nullptr) != 0) {
        std::cerr << "Error: Could not open video file " << videoFile << std::endl;
        return;
    }
    if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
        std::cerr << "Error: Could not find stream information." << std::endl;
        avformat_close_input(&fmtCtx);
        return;
    }

    int vidIdx = -1;
    AVCodec* codec = nullptr; // Renamed from 'codec' to avoid conflict with outer scope 'codec' if any
    AVCodecContext* ctx = nullptr; // Renamed from 'ctx'

    for (unsigned i = 0; i < fmtCtx->nb_streams; i++) {
        if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            vidIdx = i;
            codec = avcodec_find_decoder(fmtCtx->streams[i]->codecpar->codec_id);
            if (!codec) {
                std::cerr << "Error: Unsupported codec!" << std::endl;
                avformat_close_input(&fmtCtx);
                return;
            }
            break;
        }
    }
    if (vidIdx == -1 || !codec) {
         std::cerr << "Error: Could not find video stream or decoder." << std::endl;
        avformat_close_input(&fmtCtx);
        return;
    }

    ctx = avcodec_alloc_context3(codec);
    if (!ctx) {
        std::cerr << "Error: Could not allocate codec context." << std::endl;
        avformat_close_input(&fmtCtx);
        return;
    }
    if (avcodec_parameters_to_context(ctx, fmtCtx->streams[vidIdx]->codecpar) < 0) {
        std::cerr << "Error: Could not copy codec parameters to context." << std::endl;
        avcodec_free_context(&ctx);
        avformat_close_input(&fmtCtx);
        return;
    }

    ctx->thread_count = std::max(1u, std::thread::hardware_concurrency() / 2);
    
    AVBufferRef* hwCtxRef = nullptr; // Renamed from hwCtx to avoid confusion
    bool useHW = initializeHardwareDecoder(ctx, &hwCtxRef); // Pass address of the pointer

    if (avcodec_open2(ctx, codec, nullptr) < 0) {
        std::cerr << "Error: Could not open codec." << std::endl;
        if (hwCtxRef) av_buffer_unref(&hwCtxRef);
        avcodec_free_context(&ctx);
        avformat_close_input(&fmtCtx);
        return;
    }

    AVFrame* frame = av_frame_alloc();
    AVFrame* swFrame = nullptr; // Allocate only if needed
    if (useHW) { // If using HW acceleration, we'll need a software frame for transfer
        swFrame = av_frame_alloc();
        if (!swFrame) {
            std::cerr << "Error: Could not allocate swFrame." << std::endl;
            // Cleanup and exit
            av_frame_free(&frame);
            if (hwCtxRef) av_buffer_unref(&hwCtxRef);
            avcodec_close(ctx); // Close codec before freeing context
            avcodec_free_context(&ctx);
            avformat_close_input(&fmtCtx);
            return;
        }
    }
    
    struct SwsContext* swsCtx = nullptr;
    AVPacket* packet = av_packet_alloc();

    if (!frame || !packet) {
        std::cerr << "Error: Could not allocate frame or packet." << std::endl;
        av_frame_free(&frame);
        av_frame_free(&swFrame); // swFrame might be null, av_frame_free handles null
        av_packet_free(&packet);
        if (hwCtxRef) av_buffer_unref(&hwCtxRef);
        avcodec_close(ctx); // Close codec before freeing context
        avcodec_free_context(&ctx);
        avformat_close_input(&fmtCtx);
        return;
    }


    int gopIdx=0;
    long long carry=0; // This carry seems to be for the GOP sampling logic, not directly FFmpeg
    // long long savedCarry=0; // Renamed to avoid confusion if 'carry' is used elsewhere
    std::vector<AVPacket*> gopBuf;
    
    std::vector<std::tuple<int, int, cv::Mat>> allDecodedImages;

	auto decode_and_save = [&](const GOPSamplingInfo& info) {
        // int last = info.lastSampledOrdinal; // This is the ordinal of the last frame to be sampled in the GOP
        std::vector<int> selectedFramesIndices = info.frameIndicesToDecode; // These are 1-based indices within the GOP

        // std::cout << "Decoding GOP " << info.gopIndex << ". Frames to sample at ordinals: ";
        // for(int f_idx : selectedFramesIndices) std::cout << f_idx << " ";
        // std::cout << std::endl;

        // Iterate through the buffered packets for the current GOP
        for (int pkt_idx = 0; pkt_idx < (int)gopBuf.size(); ++pkt_idx) {
            AVPacket* current_pkt = gopBuf[pkt_idx];
            int frame_ordinal_in_gop = pkt_idx + 1; // 1-based ordinal of the frame this packet belongs to

            if (avcodec_send_packet(ctx, current_pkt) < 0) {
                // std::cerr << "Error sending packet for decoding in GOP " << info.gopIndex << ", Pkt " << pkt_idx << std::endl;
                continue; // Skip this packet
            }

            while (true) {
                int ret = avcodec_receive_frame(ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break; // Need more packets or end of stream for this GOP
                } else if (ret < 0) {
                    // std::cerr << "Error receiving frame during decoding in GOP " << info.gopIndex << ", Pkt " << pkt_idx << ": " << ffmpeg_error_string(ret) << std::endl;
                    break; 
                }

                // Successfully received a frame
                AVFrame* frameToProcess = frame; // By default, process the raw decoded frame

                if (useHW && frame->format == ctx->pix_fmt) { // ctx->pix_fmt should be the HW format, e.g., AV_PIX_FMT_CUDA
                    // The 'frame' is a hardware surface frame. Transfer it to 'swFrame' (a software frame).
                    if (av_hwframe_transfer_data(swFrame, frame, 0) < 0) {
                        std::cerr << "Error transferring HW frame to SW frame." << std::endl;
                        av_frame_unref(frame); // Unref the received HW frame
                        continue; // Skip processing this frame
                    }
                    // swFrame now contains the image data in a software format (e.g., NV12, YUV420P)
                    // Update swFrame's dimensions as they might not be set by transfer_data alone
                    swFrame->width = frame->width;
                    swFrame->height = frame->height;
                    // swFrame->format is set by av_hwframe_transfer_data or should be known
                    frameToProcess = swFrame; // Process the software frame
                }

                // Check if the current frame's ordinal (pkt_idx + 1) is in the list of frames to be sampled
                if (std::find(selectedFramesIndices.begin(), selectedFramesIndices.end(), frame_ordinal_in_gop) != selectedFramesIndices.end()) {
                    // This frame needs to be saved
                    if (!swsCtx) { // Initialize SWS context for BGR24 conversion if not already done
                        swsCtx = sws_getContext(frameToProcess->width, frameToProcess->height, (AVPixelFormat)frameToProcess->format,
                                                 frameToProcess->width, frameToProcess->height, AV_PIX_FMT_BGR24,
                                                 SWS_BICUBIC, nullptr, nullptr, nullptr);
                        if (!swsCtx) {
                            std::cerr << "Error: Could not initialize SwsContext." << std::endl;
                            // This is a critical error for saving images, might need to abort or skip all saving
                            av_frame_unref(frame); // Unref original frame
                            if (useHW && frameToProcess == swFrame) av_frame_unref(swFrame); // Unref swFrame if it was used
                            return; // Exit lambda or handle error appropriately
                        }
                    }
                    
                    cv::Mat img(frameToProcess->height, frameToProcess->width, CV_8UC3);
                    uint8_t* dst_data[1] = { img.data };
                    int dst_linesize[1] = { (int)img.step[0] };
                    sws_scale(swsCtx, frameToProcess->data, frameToProcess->linesize, 0, frameToProcess->height, dst_data, dst_linesize);
                    
                    allDecodedImages.push_back(std::make_tuple(info.gopIndex, frame_ordinal_in_gop, img.clone()));
                    // std::cout << "  Saved frame: GOP " << info.gopIndex << ", Ordinal in GOP " << frame_ordinal_in_gop << std::endl;
                }

                av_frame_unref(frame); // Unref the original decoded frame (frame)
                if (useHW && frameToProcess == swFrame) {
                    av_frame_unref(swFrame); // Unref the software frame if it was used
                }
            }
        }
        // After processing all packets in the GOP buffer, flush the decoder for any remaining frames.
        // This is important for codecs that buffer multiple frames.
        // Send NULL packet to flush
        avcodec_send_packet(ctx, nullptr);
        while (true) { // Drain any remaining frames from the decoder for this GOP
            int ret = avcodec_receive_frame(ctx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            } else if (ret < 0) {
                // std::cerr << "Error receiving flushed frame: " << ffmpeg_error_string(ret) << std::endl;
                break;
            }
            // Process flushed frame (similar logic as above)
            // This part might be tricky as we don't have a direct packet ordinal for flushed frames.
            // For GOP-based sampling, flushing is usually done at the very end of the stream,
            // or if strictly necessary, one must carefully manage frame counting.
            // Given the current sampling logic is based on packet ordinals within a GOP,
            // flushed frames might not align with 'selectedFramesIndices'.
            // For simplicity, we might ignore flushed frames in the context of this specific sampling strategy,
            // or assume they are not part of the 'selectedFramesIndices'.
            // However, for correctness, a decoder flush should be performed.
            // The impact on sampling needs careful consideration if flushed frames are to be sampled.

            // std::cout << "  Processing a flushed frame from GOP " << info.gopIndex << std::endl;
            // (Add processing logic for flushed frames if they need to be sampled,
            //  otherwise just unref them)

            av_frame_unref(frame);
            if (useHW && swFrame->data[0]) av_frame_unref(swFrame); // If swFrame was used and has data
        }
        // avcodec_flush_buffers(ctx); // This is a harder reset, typically used when seeking or changing parameters.
                                   // For end-of-GOP processing, sending a NULL packet is preferred for draining.
    };

    long long current_carry_for_gop_sampling = 0; // Initialize carry for the first GOP

    while (av_read_frame(fmtCtx, packet) >= 0) {
        if (packet->stream_index != vidIdx) {
            av_packet_unref(packet);
            continue;
        }
        bool isKeyFrame = packet->flags & AV_PKT_FLAG_KEY;

        if (isKeyFrame) {
            if (gopIdx > 0 && !gopBuf.empty()) { // Process previous GOP if it exists
                GOPSamplingInfo info = calculate_sample_indices_for_gop(
                    gopIdx, gopBuf.size(), current_carry_for_gop_sampling, interval, false);
                current_carry_for_gop_sampling = info.next_carry_over; // Update carry for next GOP
                
                if (info.samplesToTake > 0) {
                    decode_and_save(info);
                }
                for (auto& p_item : gopBuf) av_packet_free(&p_item);
                gopBuf.clear();
            }
            gopIdx++; // Increment GOP index for the new GOP starting with this keyframe
            // The 'carry' for this new GOP is what was calculated from the previous one.
        }
        
        // Buffer packet if it belongs to a valid GOP (i.e., after the first keyframe has been seen)
        if (gopIdx > 0) { 
            AVPacket* cloned_packet = av_packet_clone(packet);
            if (cloned_packet) {
                gopBuf.push_back(cloned_packet);
            } else {
                std::cerr << "Failed to clone packet, skipping." << std::endl;
            }
        }
        av_packet_unref(packet); // Unref original packet
    }

    // Process the last GOP after loop finishes (EOF)
    if (gopIdx > 0 && !gopBuf.empty()) {
        GOPSamplingInfo info = calculate_sample_indices_for_gop(
            gopIdx, gopBuf.size(), current_carry_for_gop_sampling, interval, true); // is_last_gop = true
        // No need to update carry from the last GOP
        if (info.samplesToTake > 0) {
            decode_and_save(info);
        }
        for (auto& p_item : gopBuf) av_packet_free(&p_item);
        gopBuf.clear();
    }

    // Final flush of the decoder at the end of the stream
    // avcodec_send_packet(ctx, nullptr); // Send NULL to flush decoder
    // while (avcodec_receive_frame(ctx, frame) == 0) {
        // Process any final flushed frames if necessary, similar to decode_and_save logic
        // This might be important if the sampling strategy needs to consider frames up to the very end.
        // However, the GOP-based logic might have already covered this.
    //    av_frame_unref(frame);
    //    if (useHW && swFrame->data[0]) av_frame_unref(swFrame);
    // }


    std::vector<std::tuple<int, int, cv::Mat>> finalImagesToSave = allDecodedImages;
    
    if (SAVE_IMAGE_COUNT > 0 && (int)finalImagesToSave.size() > SAVE_IMAGE_COUNT) {
        // Keep first SAVE_IMAGE_COUNT images
        finalImagesToSave.resize(SAVE_IMAGE_COUNT);
    } else if (SAVE_IMAGE_COUNT < 0 && (int)finalImagesToSave.size() > std::abs(SAVE_IMAGE_COUNT)) {
        // Keep last abs(SAVE_IMAGE_COUNT) images
        finalImagesToSave.erase(finalImagesToSave.begin(), finalImagesToSave.end() - std::abs(SAVE_IMAGE_COUNT));
    } else if (SAVE_IMAGE_COUNT == 0) { // A value of 0 means do not save any images
        finalImagesToSave.clear();
    }
    // If SAVE_IMAGE_COUNT is non-zero and abs(SAVE_IMAGE_COUNT) >= finalImagesToSave.size(), all images are kept.

    for (const auto& imageInfoTuple : finalImagesToSave) {
        int gopIndexVal = std::get<0>(imageInfoTuple);
        int frameIndexInGopVal = std::get<1>(imageInfoTuple); // This is the ordinal within the GOP
        const cv::Mat& imgToSave = std::get<2>(imageInfoTuple);
        
        std::string filename = generateTimestampFilename("frame", gopIndexVal, frameIndexInGopVal, ".jpg");
        fs::path outputPath = fs::path(outputDir) / filename;
        try {
            cv::imwrite(outputPath.string(), imgToSave);
        } catch (const cv::Exception& ex) {
            std::cerr << "OpenCV Error saving image " << outputPath.string() << ": " << ex.what() << std::endl;
        }
    }

    sws_freeContext(swsCtx);
    av_frame_free(&swFrame); // av_frame_free handles NULL pointers
    av_frame_free(&frame);
    av_packet_free(&packet);
    if (hwCtxRef) av_buffer_unref(&hwCtxRef); // Unref the HW device context
    if (ctx) {
        avcodec_close(ctx); // Close the codec context before freeing
        avcodec_free_context(&ctx);
    }
    if (fmtCtx) avformat_close_input(&fmtCtx);

    std::cout << "Processing finished. Total images saved to disk: " << finalImagesToSave.size() << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <video_file_path> <sampling_interval> [output_directory]" << std::endl;
        return -1;
    }
    // av_log_set_level(AV_LOG_DEBUG); // Enable more detailed FFmpeg logs for debugging
    av_log_set_level(AV_LOG_ERROR); // Set to AV_LOG_ERROR for less verbose output

    std::string videoFilePath = argv[1];
    int samplingInterval = 0;
    try {
        samplingInterval = std::stoi(argv[2]);
        if (samplingInterval <= 0) {
            std::cerr << "Error: Sampling interval must be a positive integer." << std::endl;
            return -1;
        }
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Error: Invalid sampling interval. Please provide an integer. " << ia.what() << std::endl;
        return -1;
    } catch (const std::out_of_range& oor) {
        std::cerr << "Error: Sampling interval out of range. " << oor.what() << std::endl;
        return -1;
    }

    std::string outputDirectory = (argc > 3) ? argv[3] : "./output_frames";
    
    processAndDecodeVideo(videoFilePath, samplingInterval, outputDirectory);
    
    return 0;
}

