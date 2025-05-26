#!/bin/bash

set -e

echo "🔧 [1/5] 设置必要路径..."
OPENCV_SRC_DIR="/home/tju/opencv_compile_workspace/opencv"
OPENCV_CONTRIB_DIR="/home/tju/opencv_compile_workspace/opencv_contrib/modules"
VIDEO_CODEC_SDK="/home/tju/workspace/Video_Codec_SDK_12.0.16"
INSTALL_PREFIX="/home/tju/opencv"
BUILD_DIR="/home/tju/opencv_compile_workspace/build"

echo "📁 [2/5] 创建头文件软链接到 /usr/local/include ..."
sudo ln -sf ${VIDEO_CODEC_SDK}/Interface/nvcuvid.h /usr/local/cuda/include/nvcuvid.h
sudo ln -sf ${VIDEO_CODEC_SDK}/Interface/cuviddec.h /usr/local/cuda/include/cuviddec.h
sudo ln -sf ${VIDEO_CODEC_SDK}/Interface/nvEncodeAPI.h /usr/local/cuda/include/nvEncodeAPI.h

echo "📦 [3/5] 安装并确认动态库 ..."
sudo apt update
sudo apt install -y nvidia-utils-550

# 检查是否存在库
if [[ ! -f "/usr/lib/x86_64-linux-gnu/libnvcuvid.so" ]] || [[ ! -f "/usr/lib/x86_64-linux-gnu/libnvidia-encode.so" ]]; then
  echo "❌ 缺少 NVIDIA 解码/编码库（libnvcuvid.so 或 libnvidia-encode.so）"
  exit 1
fi

echo "🧹 [4/5] 清理旧的 build 目录 ..."
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
      -D OPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB_DIR} \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=8.6 \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D WITH_CUBLAS=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_NVCUVID=ON \
      -D WITH_NVCUVENC=ON \
      -D BUILD_opencv_cudacodec=ON \
      -D CUDA_NVCC_FLAGS="-I/usr/local/include" \
      -D CMAKE_LINK_FLAGS="-L/usr/lib/x86_64-linux-gnu -L/home/tju/workspace/Video_Codec_SDK_12.0.16/Lib/linux" \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      ${OPENCV_SRC_DIR}

