#!/usr/bin/env bash
set -o pipefail

# 自动检测依赖环境
PKG_OK=true

check_pkg() {
    if ! pkg-config --exists "$1"; then
        echo "[ERROR] Missing pkg: $1" >&2
        PKG_OK=false
    fi
}

# 检查必要依赖
check_pkg libavformat
check_pkg libavcodec
check_pkg libavutil
check_pkg libswscale
check_pkg opencv4

if ! $PKG_OK; then
    echo "[ERROR] Required packages not found, aborting." >&2
    exit 1
fi

# 源文件列表
SRC_FILES="main.cpp video_processor.cpp packet_decoder.cpp"
OUT_FILE="skip_frame"

# 获取编译参数
CXX_FLAGS="-std=c++17 -Wall -Wno-deprecated-declarations"
FFMPEG_CFLAGS=$(pkg-config --cflags libavformat libavcodec libavutil libswscale)
FFMPEG_LIBS=$(pkg-config --libs libavformat libavcodec libavutil libswscale)
OPENCV_LIBS=$(pkg-config --libs opencv4)

# CUDA（可选支持）
CUDA_FLAGS=""
if nvcc --version &>/dev/null; then
    CUDA_FLAGS="-lcuda -lcudart"
fi

# 编译
g++ $CXX_FLAGS $SRC_FILES -o "$OUT_FILE" \
    $FFMPEG_CFLAGS \
    $FFMPEG_LIBS $OPENCV_LIBS $CUDA_FLAGS \
    -lpthread 2> .compile_log

if [ $? -ne 0 ]; then
    echo "[ERROR] Compilation failed:"
    cat .compile_log >&2
    exit 1
fi

rm -f .compile_log
