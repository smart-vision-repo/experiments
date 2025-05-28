#!/usr/bin/env bash

set -e  # 一旦有命令失败就退出
set -o pipefail

SRC_FILES="main.cpp video_processor.cpp packet_decoder.cpp"
OUT_FILE="app_video_processor"

# FFmpeg & OpenCV 常见路径 (Homebrew 安装在 M1 Mac 下)
INCLUDE_FLAGS="-I/opt/homebrew/include -I/opt/homebrew/include/opencv4"
LIB_FLAGS="-L/opt/homebrew/lib"
FFMPEG_LIBS="-lavformat -lavcodec -lavutil -lswscale"
OPENCV_LIBS=$(pkg-config --libs opencv4 2>/dev/null || echo "-lopencv_core -lopencv_imgproc -lopencv_highgui")

# 编译命令
g++ -std=c++17 $SRC_FILES -o $OUT_FILE \
    $INCLUDE_FLAGS $LIB_FLAGS \
    $FFMPEG_LIBS $OPENCV_LIBS \
    -lpthread -Wall -Wno-deprecated-declarations \
    2> .compile_log

# 如果失败才输出错误
if [ $? -ne 0 ]; then
    cat .compile_log
    echo "[ERROR] Compilation failed."
    exit 1
fi

rm -f .compile_log