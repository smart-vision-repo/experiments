#include <iostream>
#include <string>
#include <stdexcept> // For std::stoi exception handling

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <video_file> <interval>" << std::endl;
        return -1;
    }

    int interval = 0;
    try {
        interval = std::stoi(argv[2]);
        if (interval <= 0) {
            std::cout << "Interval must be a positive integer." << std::endl;
            return -1;
        }
    } catch (const std::invalid_argument& ia) {
        std::cout << "Invalid interval argument: " << argv[2] << std::endl;
        return -1;
    } catch (const std::out_of_range& oor) {
        std::cout << "Interval argument out of range: " << argv[2] << std::endl;
        return -1;
    }

    AVFormatContext* formatContext = nullptr;
    if (avformat_open_input(&formatContext, argv[1], nullptr, nullptr) < 0) {
        std::cout << "Could not open video file: " << argv[1] << std::endl;
        return -1;
    }

    if (avformat_find_stream_info(formatContext, nullptr) < 0) {
        std::cout << "Could not get stream info" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    int videoStream = -1;
    for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStream = i;
            break;
        }
    }

    if (videoStream == -1) {
        std::cout << "No video stream found" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    AVPacket* packet = av_packet_alloc();
    if (!packet) {
        std::cout << "Could not allocate AVPacket" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    long long total_video_frames_processed_summary = 0;
    int total_i_frames_summary = 0;

    int current_gop_idx_for_a = 0; 
    int frames_in_current_gop_for_b = 0; 

    long long carry_over_frames_for_sampling = 0; 
    long long saved_initial_carry_for_gop_calc = 0; 

    long long total_hitted_frames_sum_c = 0;
    long long total_discarded_frames_sum_e = 0;
    // New summary variables for d and b
    long long total_last_sampled_ordinals_sum_d = 0;
    long long total_frames_in_gops_sum_b = 0;


    std::cout << "a (GOP Index),b (Frames in GOP),c (Samples in GOP),d (Last Sampled Ordinal in GOP),e (Remaining Frames in GOP after d)" << std::endl;

    while (av_read_frame(formatContext, packet) >= 0) {
        if (packet->stream_index == videoStream) {
            total_video_frames_processed_summary++;
            bool isKeyFrame = (packet->flags & AV_PKT_FLAG_KEY);

            if (isKeyFrame) {
                if (current_gop_idx_for_a > 0) { 
                    int b_val = frames_in_current_gop_for_b;
                    long long pool = saved_initial_carry_for_gop_calc + b_val;
                    int c_val = 0;
                    int d_val = 0;
                    int e_val = b_val; 

                    if (interval > 0) { 
                        c_val = pool / interval;
                        if (c_val > 0) {
                            long long first_sample_gop_ordinal = interval - saved_initial_carry_for_gop_calc;
                            d_val = first_sample_gop_ordinal + (long long)(c_val - 1) * interval;
                        }
                    }
                    
                    if (d_val > 0 && d_val <= b_val) {
                        e_val = b_val - d_val;
                    } else { 
                        d_val = 0; 
                        e_val = b_val;
                    }
                    
                    std::cout << current_gop_idx_for_a << ","
                              << b_val << ","
                              << c_val << ","
                              << d_val << ","
                              << e_val << std::endl;

                    total_hitted_frames_sum_c += c_val;
                    total_discarded_frames_sum_e += e_val;
                    total_last_sampled_ordinals_sum_d += d_val; // Sum d
                    total_frames_in_gops_sum_b += b_val;      // Sum b

                    carry_over_frames_for_sampling = pool % interval; 
                }

                total_i_frames_summary++;
                current_gop_idx_for_a++;
                frames_in_current_gop_for_b = 1; 
                saved_initial_carry_for_gop_calc = carry_over_frames_for_sampling; 
            } else {
                if (current_gop_idx_for_a > 0) { 
                    frames_in_current_gop_for_b++;
                }
            }
        }
        av_packet_unref(packet);
    }

    if (current_gop_idx_for_a > 0 && frames_in_current_gop_for_b > 0) {
        int b_val = frames_in_current_gop_for_b;
        long long pool = saved_initial_carry_for_gop_calc + b_val;
        int c_val = 0;
        int d_val = 0;
        int e_val = b_val;

        if (interval > 0) {
             c_val = pool / interval;
            if (c_val > 0) {
                long long first_sample_gop_ordinal = interval - saved_initial_carry_for_gop_calc;
                d_val = first_sample_gop_ordinal + (long long)(c_val - 1) * interval;
            }
        }

        if (d_val > 0 && d_val <= b_val) {
            e_val = b_val - d_val;
        } else {
            d_val = 0;
            e_val = b_val;
        }

        std::cout << current_gop_idx_for_a << ","
                  << b_val << ","
                  << c_val << ","
                  << d_val << ","
                  << e_val << std::endl;
        
        total_hitted_frames_sum_c += c_val;
        total_discarded_frames_sum_e += e_val;
        total_last_sampled_ordinals_sum_d += d_val; // Sum d for last GOP
        total_frames_in_gops_sum_b += b_val;      // Sum b for last GOP
    }

    av_packet_free(&packet);
    avformat_close_input(&formatContext);

    // --- Statistics ---
    std::cout << "--- Statistics ---" << std::endl;
    std::cout << "Total I-frames: " << total_i_frames_summary << std::endl;
    std::cout << "Total video frames processed: " << total_video_frames_processed_summary << std::endl;
    std::cout << "Total hitted frames (sum of 'c' values): " << total_hitted_frames_sum_c << std::endl;
    std::cout << "Total frames remaining in GOPs after last sample (sum of 'e' values): " << total_discarded_frames_sum_e << std::endl;
    std::cout << "Sum of all 'd' values (sum of last sampled frame ordinals in GOPs): " << total_last_sampled_ordinals_sum_d << std::endl;
    std::cout << "Sum of all 'b' values (total frames in identified GOPs): " << total_frames_in_gops_sum_b << std::endl;
    std::cout << "Verification: sum(d) + sum(e) = " << (total_last_sampled_ordinals_sum_d + total_discarded_frames_sum_e) << std::endl;
    std::cout << "(This sum should equal sum(b). If sum(b) is less than 'Total video frames processed',"
              << " it may be due to frames existing before the first I-frame)." << std::endl;

    return 0;
}
