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


AVPixelFormat getHWFormatCallback(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
    
    if (!ctx || !ctx->hw_device_ctx || !ctx->hw_device_ctx->data) {
        std::cerr << "getHWFormatCallback: ERROR - Missing hardware device context in AVCodecContext." << std::endl;
        return AV_PIX_FMT_NONE;
    }

    
    
    
    AVHWDeviceContext *hw_device_ctx_data = (AVHWDeviceContext*)ctx->hw_device_ctx->data;
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




int initializeHardwareDecoder(AVCodecContext* codecContext, AVBufferRef** hwDeviceCtx) {
    
    if (*hwDeviceCtx != nullptr) {
        av_buffer_unref(hwDeviceCtx); 
        *hwDeviceCtx = nullptr;
    }

    const char* hwAccel[] = {"cuda", "vaapi", "vdpau", "qsv", "d3d11va", "videotoolbox", nullptr};
    for (int i = 0; hwAccel[i]; i++) {
        enum AVHWDeviceType type = av_hwdevice_find_type_by_name(hwAccel[i]);
        if (type == AV_HWDEVICE_TYPE_NONE) {
            
            
            continue;
        }

        
        
        int ret = av_hwdevice_ctx_create(hwDeviceCtx, type, 0, nullptr, 0);
        if (ret >= 0) {
            
            
            if (codecContext->hw_device_ctx) { 
                 av_buffer_unref(&codecContext->hw_device_ctx);
            }
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

GOPSamplingInfo calculate_sample_indices_for_gop(int gop_idx, int b_val, long long initial_carry, int interval, bool is_last_gop) {
    GOPSamplingInfo result;
    result.gopIndex = gop_idx;
    result.framesInGop = b_val;
    long long pool = initial_carry + b_val;
    int c_val = (interval>0) ? pool/interval : 0;
    int d_val = 0;
    if (c_val>0) {
        long long first = interval - initial_carry;
        if (first<=0) first=1; 
        
        if (b_val >= first) { 
             d_val = first + (long long)((c_val-1)>0?(c_val-1):0)*interval; 
             
             if (d_val > b_val) {
                 
                 if (b_val < first) { 
                     d_val = 0;
                 } else {
                     d_val = first + (long long)((b_val - first) / interval) * interval;
                 }
             }
        } else {
            d_val = 0; 
        }

        if (d_val < 1) d_val = 0; 
    }

    result.lastSampledOrdinal = d_val;
    
    if (d_val > 0 && d_val >= (interval - initial_carry)) {
         result.samplesToTake = ((d_val - (interval - initial_carry)) / interval) + 1;
    } else {
        result.samplesToTake = 0;
    }
    
    for (int i=0; i<result.samplesToTake; i++) {
        result.frameIndicesToDecode.push_back((interval-initial_carry) + i*interval);
    }
    result.remainingFramesAfterD = (d_val>0)? (b_val-d_val):b_val; 
    result.next_carry_over = is_last_gop?0:(pool%interval);
    if (pool < interval && !is_last_gop) { 
        result.next_carry_over = pool;
    }


    
    
    
    
    
    
    
    
    

    return result;
}

int totalFrames = 0;
int hwDecodedFrames = 0;
int swDecodedFrames = 0;

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
    AVCodec* codec = nullptr; 
    AVCodecContext* ctx = nullptr; 

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
    
    AVBufferRef* hwCtxRef = nullptr; 
    bool useHW = initializeHardwareDecoder(ctx, &hwCtxRef); 

    if (avcodec_open2(ctx, codec, nullptr) < 0) {
        std::cerr << "Error: Could not open codec." << std::endl;
        if (hwCtxRef) av_buffer_unref(&hwCtxRef);
        avcodec_free_context(&ctx);
        avformat_close_input(&fmtCtx);
        return;
    }

    AVFrame* frame = av_frame_alloc();
    AVFrame* swFrame = nullptr; 
    if (useHW) { 
        swFrame = av_frame_alloc();
        if (!swFrame) {
            std::cerr << "Error: Could not allocate swFrame." << std::endl;
            
            av_frame_free(&frame);
            if (hwCtxRef) av_buffer_unref(&hwCtxRef);
            avcodec_close(ctx); 
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
        av_frame_free(&swFrame); 
        av_packet_free(&packet);
        if (hwCtxRef) av_buffer_unref(&hwCtxRef);
        avcodec_close(ctx); 
        avcodec_free_context(&ctx);
        avformat_close_input(&fmtCtx);
        return;
    }


    int gopIdx=0;
    long long carry=0; 
    
    std::vector<AVPacket*> gopBuf;
    
    std::vector<std::tuple<int, int, cv::Mat>> allDecodedImages;

	auto decode_and_save = [&](const GOPSamplingInfo& info) {
        
        std::vector<int> selectedFramesIndices = info.frameIndicesToDecode; 

        
        
        

        
        for (int pkt_idx = 0; pkt_idx < (int)gopBuf.size(); ++pkt_idx) {
            AVPacket* current_pkt = gopBuf[pkt_idx];
            int frame_ordinal_in_gop = pkt_idx + 1; 

            if (avcodec_send_packet(ctx, current_pkt) < 0) {
                
                continue; 
            }

            while (true) {
                int ret = avcodec_receive_frame(ctx, frame);
		

                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break; 
                } else if (ret < 0) {
                    
                    break; 
                }
                AVFrame* frameToProcess = frame; 
						 
		++totalFrames;
		if (useHW && frame->format == ctx->pix_fmt) {
		    ++hwDecodedFrames;
		} else {
		    ++swDecodedFrames;
		}

                if (useHW && frame->format == ctx->pix_fmt) { 
                    
                    if (av_hwframe_transfer_data(swFrame, frame, 0) < 0) {
                        std::cerr << "Error transferring HW frame to SW frame." << std::endl;
                        av_frame_unref(frame); 
                        continue; 
                    }
                    
                    
                    swFrame->width = frame->width;
                    swFrame->height = frame->height;
                    
                    frameToProcess = swFrame; 
                }

                
                if (std::find(selectedFramesIndices.begin(), selectedFramesIndices.end(), frame_ordinal_in_gop) != selectedFramesIndices.end()) {
                    
                    if (!swsCtx) { 
                        swsCtx = sws_getContext(frameToProcess->width, frameToProcess->height, (AVPixelFormat)frameToProcess->format,
                                                 frameToProcess->width, frameToProcess->height, AV_PIX_FMT_BGR24,
                                                 SWS_BICUBIC, nullptr, nullptr, nullptr);
                        if (!swsCtx) {
                            std::cerr << "Error: Could not initialize SwsContext." << std::endl;
                            
                            av_frame_unref(frame); 
                            if (useHW && frameToProcess == swFrame) av_frame_unref(swFrame); 
                            return; 
                        }
                    }
                    
                    cv::Mat img(frameToProcess->height, frameToProcess->width, CV_8UC3);
                    uint8_t* dst_data[1] = { img.data };
                    int dst_linesize[1] = { (int)img.step[0] };
                    sws_scale(swsCtx, frameToProcess->data, frameToProcess->linesize, 0, frameToProcess->height, dst_data, dst_linesize);
                    
                    allDecodedImages.push_back(std::make_tuple(info.gopIndex, frame_ordinal_in_gop, img.clone()));
                    
                }

                av_frame_unref(frame); 
                if (useHW && frameToProcess == swFrame) {
                    av_frame_unref(swFrame); 
                }
            }
        }
        
        
        
        
        avcodec_send_packet(ctx, nullptr);
        while (true) { 
            int ret = avcodec_receive_frame(ctx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            } else if (ret < 0) {
                
                break;
            }
            
            
            
            
            
            
            
            

            
            
            
            

            av_frame_unref(frame);
            if (useHW && swFrame->data[0]) av_frame_unref(swFrame); 
        }
        
                                   
    };

    long long current_carry_for_gop_sampling = 0; 

    while (av_read_frame(fmtCtx, packet) >= 0) {
        if (packet->stream_index != vidIdx) {
            av_packet_unref(packet);
            continue;
        }
        bool isKeyFrame = packet->flags & AV_PKT_FLAG_KEY;

        if (isKeyFrame) {
            if (gopIdx > 0 && !gopBuf.empty()) { 
                GOPSamplingInfo info = calculate_sample_indices_for_gop(
                    gopIdx, gopBuf.size(), current_carry_for_gop_sampling, interval, false);
                current_carry_for_gop_sampling = info.next_carry_over; 
                
                if (info.samplesToTake > 0) {
                    decode_and_save(info);
                }
                for (auto& p_item : gopBuf) av_packet_free(&p_item);
                gopBuf.clear();
            }
            gopIdx++; 
            
        }
        
        
        if (gopIdx > 0) { 
            AVPacket* cloned_packet = av_packet_clone(packet);
            if (cloned_packet) {
                gopBuf.push_back(cloned_packet);
            } else {
                std::cerr << "Failed to clone packet, skipping." << std::endl;
            }
        }
        av_packet_unref(packet); 
    }

    
    if (gopIdx > 0 && !gopBuf.empty()) {
        GOPSamplingInfo info = calculate_sample_indices_for_gop(
            gopIdx, gopBuf.size(), current_carry_for_gop_sampling, interval, true); 
        
        if (info.samplesToTake > 0) {
            decode_and_save(info);
        }
        for (auto& p_item : gopBuf) av_packet_free(&p_item);
        gopBuf.clear();
    }

    
    
    
        
        
        
    
    
    


    std::vector<std::tuple<int, int, cv::Mat>> finalImagesToSave = allDecodedImages;
    
    if (SAVE_IMAGE_COUNT > 0 && (int)finalImagesToSave.size() > SAVE_IMAGE_COUNT) {
        
        finalImagesToSave.resize(SAVE_IMAGE_COUNT);
    } else if (SAVE_IMAGE_COUNT < 0 && (int)finalImagesToSave.size() > std::abs(SAVE_IMAGE_COUNT)) {
        
        finalImagesToSave.erase(finalImagesToSave.begin(), finalImagesToSave.end() - std::abs(SAVE_IMAGE_COUNT));
    } else if (SAVE_IMAGE_COUNT == 0) { 
        finalImagesToSave.clear();
    }
    

    for (const auto& imageInfoTuple : finalImagesToSave) {
        int gopIndexVal = std::get<0>(imageInfoTuple);
        int frameIndexInGopVal = std::get<1>(imageInfoTuple); 
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
    av_frame_free(&swFrame); 
    av_frame_free(&frame);
    av_packet_free(&packet);
    if (hwCtxRef) av_buffer_unref(&hwCtxRef); 
    if (ctx) {
        avcodec_close(ctx); 
        avcodec_free_context(&ctx);
    }
    if (fmtCtx) avformat_close_input(&fmtCtx);

    std::cout << "Processing finished. Total images saved to disk: " << finalImagesToSave.size() << std::endl;
    std::cout << "Total decoded frames: " << totalFrames << std::endl;
    std::cout << "Hardware decoded frames: " << hwDecodedFrames << std::endl;
    std::cout << "Software decoded frames: " << swDecodedFrames << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <video_file_path> <sampling_interval> [output_directory]" << std::endl;
        return -1;
    }
    
    av_log_set_level(AV_LOG_ERROR); 

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
