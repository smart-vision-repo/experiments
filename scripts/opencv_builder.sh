#!/bin/bash
set -e # 如果任何命令失败，立即退出脚本

# --- 用户可配置变量 ---
OPENCV_VERSION="4.9.0" 
OPENCV_CONTRIB_VERSION="${OPENCV_VERSION}"
CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda-12.2" # !!! 非常重要：确保这是你为OpenCV编译选择的CUDA版本路径 !!!
VIDEO_CODEC_SDK_DIR="/opt/Video_Codec_SDK_12.1.14" # !!! 请确认此路径绝对正确 !!!
INSTALL_DIR="/opt/opencv"
NUM_JOBS=$(nproc)
PERSISTENT_WORK_DIR="${HOME}/installations/opencv" 
# --- 用户可配置变量结束 ---

# 设置 Python 路径
PY_EXECUTABLE=$(which python3)
PY_VERSION=$(python3 -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
PY_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
PY_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY') or sysconfig.get_config_var('LIBRARY')))")
PY_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
PY_NUMPY_INCLUDE=$(python3 -c "import numpy; print(numpy.get_include())")

echo "--- 配置信息 ---"
echo "OpenCV 版本: ${OPENCV_VERSION}"
echo "OpenCV Contrib 版本: ${OPENCV_CONTRIB_VERSION}"
echo "CUDA Toolkit 路径: ${CUDA_TOOLKIT_ROOT_DIR}"
echo "Video Codec SDK 路径: ${VIDEO_CODEC_SDK_DIR}"
echo "安装目录: ${INSTALL_DIR}"
echo "Make 并行作业数: ${NUM_JOBS}"
echo "固定工作目录 (用于源码和构建): ${PERSISTENT_WORK_DIR}"
echo "---------------------"

if [ ! -d "${CUDA_TOOLKIT_ROOT_DIR}" ] || [ ! -f "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" ]; then
    echo "错误：CUDA_TOOLKIT_ROOT_DIR ('${CUDA_TOOLKIT_ROOT_DIR}') 无效，或者其下缺少 'bin/nvcc'。"
    exit 1
fi
if [ ! -d "${VIDEO_CODEC_SDK_DIR}" ]; then
    echo "警告：VIDEO_CODEC_SDK_DIR ('${VIDEO_CODEC_SDK_DIR}') 目录不存在。CMake阶段可能会失败。"
fi

read -p "请仔细核对以上路径和版本。按 Enter键 继续，或按 Ctrl+C 中止。"

mkdir -p "${PERSISTENT_WORK_DIR}"
cd "${PERSISTENT_WORK_DIR}" 

OPENCV_ZIP_FILENAME="/home/tju/downloads/opencv-${OPENCV_VERSION}.zip"
OPENCV_SRC_FOLDERNAME="opencv-${OPENCV_VERSION}"

if [ ! -f "${OPENCV_ZIP_FILENAME}" ]; then
    wget -O "${OPENCV_ZIP_FILENAME}" "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip"
fi
if [ ! -d "${OPENCV_SRC_FOLDERNAME}" ]; then
    unzip -q "${OPENCV_ZIP_FILENAME}" 
fi

OPENCV_CONTRIB_ZIP_FILENAME="/home/tju/downloads/opencv_contrib-${OPENCV_CONTRIB_VERSION}.zip"
OPENCV_CONTRIB_SRC_FOLDERNAME="opencv_contrib-${OPENCV_CONTRIB_VERSION}"

if [ ! -f "${OPENCV_CONTRIB_ZIP_FILENAME}" ]; then
    wget -O "${OPENCV_CONTRIB_ZIP_FILENAME}" "https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_CONTRIB_VERSION}.zip"
fi
if [ ! -d "${OPENCV_CONTRIB_SRC_FOLDERNAME}" ]; then
    unzip -q "${OPENCV_CONTRIB_ZIP_FILENAME}" 
fi

cd "${PERSISTENT_WORK_DIR}/${OPENCV_SRC_FOLDERNAME}"
mkdir -p build 
cd build       

rm -f CMakeCache.txt
rm -rf CMakeFiles/

OPENCV_CONTRIB_MODULES_ABS_PATH="${PERSISTENT_WORK_DIR}/${OPENCV_CONTRIB_SRC_FOLDERNAME}/modules"

VIDEO_CODEC_SDK_LIB_DIR_FINAL="${VIDEO_CODEC_SDK_DIR}/Lib/Linux/stubs/x86_64"

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
      -D OPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB_MODULES_ABS_PATH} \
      -D CMAKE_INCLUDE_PATH="${VIDEO_CODEC_SDK_DIR}/Interface" \
      -D CMAKE_LIBRARY_PATH="${VIDEO_CODEC_SDK_LIB_DIR_FINAL}" \
      -D CUDA_VIDEO_INCLUDE_DIR="${VIDEO_CODEC_SDK_DIR}/Interface" \
      -D CUDA_CUDART_LIBRARY="${VIDEO_CODEC_SDK_LIB_DIR_FINAL}/libnvcuvid.so" \
      -D WITH_CUDA=ON \
      -D CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR} \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D WITH_TENSORRT=ON \
      -D OPENCV_DNN_TENSORRT=ON \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_NVCUVID=ON \
      -D WITH_NVCUVENC=OFF \
      -D WITH_FFMPEG=ON \
      -D WITH_GSTREAMER=ON \
      -D OPENCV_ENABLE_NONFREE=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_DOCS=OFF \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_FOUND=ON \
      -D PYTHON3_EXECUTABLE=${PY_EXECUTABLE} \
      -D PYTHON3_INCLUDE_DIR=${PY_INCLUDE_DIR} \
      -D PYTHON3_LIBRARY=${PY_LIBRARY} \
      -D PYTHON3_NUMPY_INCLUDE_DIRS=${PY_NUMPY_INCLUDE} \
      -D PYTHON3_PACKAGES_PATH=${PY_PACKAGES_PATH} \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D WITH_IPP=OFF \
      -D WITH_GTK=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=OFF \
      -D BUILD_opencv_wechat_qrcode=OFF \
      -D BUILD_opencv_calib3d=OFF \
      -D BUILD_opencv_features2d=ON \
      -D BUILD_opencv_flann=ON \
      -D BUILD_opencv_ml=OFF \
      -D BUILD_opencv_objdetect=OFF \
      -D BUILD_opencv_photo=OFF \
      -D BUILD_opencv_stitching=OFF \
      -D BUILD_opencv_video=OFF \
      -D BUILD_opencv_gapi=OFF \
      -D OPENCV_FORCE_DISABLE_DOWNLOADS=ON \
      -D BUILD_opencv_shape=OFF \
      -D BUILD_opencv_superres=OFF \
      -D BUILD_opencv_videostab=OFF \
      -D BUILD_opencv_viz=OFF \
      -D BUILD_opencv_aruco=OFF \
      -D BUILD_opencv_bgsegm=OFF \
      -D BUILD_opencv_bioinspired=OFF \
      -D BUILD_opencv_ccalib=OFF \
      -D BUILD_opencv_datasets=OFF \
      -D BUILD_opencv_dpm=OFF \
      -D BUILD_opencv_face=OFF \
      -D BUILD_opencv_freetype=OFF \
      -D BUILD_opencv_fuzzy=OFF \
      -D BUILD_opencv_hfs=OFF \
      -D BUILD_opencv_img_hash=OFF \
      -D BUILD_opencv_intensity_transform=OFF \
      -D BUILD_opencv_line_descriptor=OFF \
      -D BUILD_opencv_mcc=OFF \
      -D BUILD_opencv_optflow=OFF \
      -D BUILD_opencv_phase_unwrapping=OFF \
      -D BUILD_opencv_plot=OFF \
      -D BUILD_opencv_quality=OFF \
      -D BUILD_opencv_reg=OFF \
      -D BUILD_opencv_rgbd=OFF \
      -D BUILD_opencv_saliency=OFF \
      -D BUILD_opencv_stereo=OFF \
      -D BUILD_opencv_structured_light=OFF \
      -D BUILD_opencv_surface_matching=OFF \
      -D BUILD_opencv_text=OFF \
      -D BUILD_opencv_tracking=OFF \
      -D BUILD_opencv_xfeatures2d=OFF \
      -D BUILD_opencv_ximgproc=OFF \
      -D BUILD_opencv_xobjdetect=OFF \
      -D BUILD_opencv_xphoto=OFF \
      .. 

make -j${NUM_JOBS}
sudo make install

