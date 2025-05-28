compile() {
    local output_name="$1"
    shift
    local sources="$*"

    local opencv_cflags=$(pkg-config --cflags opencv4 2>/dev/null)
    local opencv_libs=$(pkg-config --libs opencv4 2>/dev/null)

    local ffmpeg_cflags=$(pkg-config --cflags libavformat libavcodec libswscale libavutil 2>/dev/null)
    local ffmpeg_libs=$(pkg-config --libs libavformat libavcodec libswscale libavutil 2>/dev/null)

    local cuda_cflags="-I/usr/local/cuda-12.2/targets/x86_64-linux/include"
    local cuda_ldflags="-L/usr/local/cuda-12.2/targets/x86_64-linux/lib"
    local cuda_link_libs="-lcudart"

    g++ -std=c++17 -o "$output_name" $sources \
        $opencv_cflags $ffmpeg_cflags $cuda_cflags \
        -DHAVE_CUDA=1 \
        $cuda_ldflags \
        $opencv_libs $ffmpeg_libs $cuda_link_libs \
        -pthread -Wno-deprecated-declarations -g

    if [ $? -ne 0 ]; then
        echo -e "\033[0;31m[ERROR]\033[0m Compilation failed."
        return 1
    else
        echo -e "\033[0;32m[INFO]\033[0m Compilation succeeded. Executable: ./$output_name"
    fi
    return 0
}

main() {
    compile "skip_frame" "main.cpp video_processor.cpp packet_decoder.cpp" || exit 1
}

main "$@"

