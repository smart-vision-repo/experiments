#!/bin/bash
set -e # 如果任何命令失败，立即退出脚本
rm -rf /home/tju/opencv_compile_workspace/opencv_4.9.0
rm -rf /home/tju/opencv_compile_workspace/opencv_contrib-4.9.0/

# --- 用户可配置变量 ---
OPENCV_VERSION="4.9.0" # 指定你希望编译的 OpenCV 版本 (请检查官网获取最新稳定版)
# Contrib 模块通常与 OpenCV 主版本相同
OPENCV_CONTRIB_VERSION="${OPENCV_VERSION}"

# 你的 CUDA Toolkit 安装路径
CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda-12.2" # 如果你的 CUDA 不在此处，请修改，例如 /usr/local/cuda-12.4

# !!! 已更新为你提供的 Video Codec SDK 路径 !!!
VIDEO_CODEC_SDK_DIR="/home/tju/downloads/Video_Codec_SDK_12.2.72"

# 此自定义 OpenCV 构建的安装目录
INSTALL_DIR="/home/tju/opencv"

# 用于 make 的并行作业数 (例如，CPU 核心数)
NUM_JOBS=$(nproc)

# 使用一个固定的工作目录来保存下载和解压的文件
PERSISTENT_WORK_DIR="${HOME}/opencv_compile_workspace" # 你可以按需修改这个目录名
# --- 用户可配置变量结束 ---

echo "--- 配置信息 ---"
echo "OpenCV 版本: ${OPENCV_VERSION}"
echo "OpenCV Contrib 版本: ${OPENCV_CONTRIB_VERSION}"
echo "CUDA Toolkit 路径: ${CUDA_TOOLKIT_ROOT_DIR}"
echo "Video Codec SDK 路径: ${VIDEO_CODEC_SDK_DIR}"
echo "安装目录: ${INSTALL_DIR}"
echo "Make 并行作业数: ${NUM_JOBS}"
echo "固定工作目录 (用于源码和构建): ${PERSISTENT_WORK_DIR}"
echo "---------------------"

# 验证 Video Codec SDK 路径是否有效 (基本检查)
if [ ! -d "${VIDEO_CODEC_SDK_DIR}" ] || [ ! -d "${VIDEO_CODEC_SDK_DIR}/Interface" ] || [ ! -d "${VIDEO_CODEC_SDK_DIR}/Lib/linux/stubs" ]; then
    echo "错误：VIDEO_CODEC_SDK_DIR ('${VIDEO_CODEC_SDK_DIR}') 无效，或者其下缺少 'Interface' 或 'Lib/linux/stubs' 子目录。"
    echo "请确认路径正确，并且 SDK 已正确解压。"
    exit 1
fi

read -p "请仔细核对以上路径和版本。按 Enter键 继续，或按 Ctrl+C 中止。"

# --- 准备固定工作空间 ---
echo ">>> 准备固定工作空间: ${PERSISTENT_WORK_DIR}"
mkdir -p "${PERSISTENT_WORK_DIR}"
cd "${PERSISTENT_WORK_DIR}" # 关键：后续下载和解压操作都在这个固定目录下进行

# --- 下载 OpenCV 和 OpenCV Contrib (如果本地不存在) ---
echo ">>> 检查/下载 OpenCV 和 Contrib 模块..."

# OpenCV 主模块
OPENCV_ZIP_FILENAME="opencv-${OPENCV_VERSION}.zip"
OPENCV_SRC_FOLDERNAME="opencv-${OPENCV_VERSION}"

if [ ! -f "${OPENCV_ZIP_FILENAME}" ]; then
    echo ">>> 下载 OpenCV ${OPENCV_VERSION}..."
    wget -O "${OPENCV_ZIP_FILENAME}" "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip"
else
    echo "OpenCV zip 文件 (${OPENCV_ZIP_FILENAME}) 已存在于 ${PERSISTENT_WORK_DIR}，跳过下载。"
fi

if [ ! -d "${OPENCV_SRC_FOLDERNAME}" ]; then
    echo ">>> 解压 OpenCV ${OPENCV_VERSION}..."
    unzip -q "${OPENCV_ZIP_FILENAME}" # -q for quiet
else
    echo "OpenCV 源目录 (${OPENCV_SRC_FOLDERNAME}) 已存在于 ${PERSISTENT_WORK_DIR}，跳过解压。"
fi

# OpenCV Contrib 模块
OPENCV_CONTRIB_ZIP_FILENAME="opencv_contrib-${OPENCV_CONTRIB_VERSION}.zip"
OPENCV_CONTRIB_SRC_FOLDERNAME="opencv_contrib-${OPENCV_CONTRIB_VERSION}"

if [ ! -f "${OPENCV_CONTRIB_ZIP_FILENAME}" ]; then
    echo ">>> 下载 OpenCV Contrib ${OPENCV_CONTRIB_VERSION}..."
    wget -O "${OPENCV_CONTRIB_ZIP_FILENAME}" "https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_CONTRIB_VERSION}.zip"
else
    echo "OpenCV Contrib zip 文件 (${OPENCV_CONTRIB_ZIP_FILENAME}) 已存在于 ${PERSISTENT_WORK_DIR}，跳过下载。"
fi

if [ ! -d "${OPENCV_CONTRIB_SRC_FOLDERNAME}" ]; then
    echo ">>> 解压 OpenCV Contrib ${OPENCV_CONTRIB_VERSION}..."
    unzip -q "${OPENCV_CONTRIB_ZIP_FILENAME}" # -q for quiet
else
    echo "OpenCV Contrib 源目录 (${OPENCV_CONTRIB_SRC_FOLDERNAME}) 已存在于 ${PERSISTENT_WORK_DIR}，跳过解压。"
fi

# --- 创建并进入构建目录 (在主 OpenCV 源码文件夹内部) ---
echo ">>> 创建/进入构建目录..."
cd "${PERSISTENT_WORK_DIR}/${OPENCV_SRC_FOLDERNAME}" # 进入解压后的 OpenCV 主源码目录
mkdir -p build # 在源码目录内创建 build 文件夹
cd build       # 进入 build 文件夹

# --- 清理旧的 CMake 缓存 (确保新配置生效) ---
echo ">>> 清理旧的 CMake 缓存 (如果存在)..."
if [ -f "CMakeCache.txt" ]; then
    rm -f CMakeCache.txt
fi
if [ -d "CMakeFiles/" ]; then
    rm -rf CMakeFiles/
fi

# --- 运行 CMake 配置 ---
echo ">>> 配置 CMake..."
OPENCV_CONTRIB_MODULES_ABS_PATH="${PERSISTENT_WORK_DIR}/${OPENCV_CONTRIB_SRC_FOLDERNAME}/modules"

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
      -D CMAKE_PREFIX_PATH="${VIDEO_CODEC_SDK_DIR};${CUDA_TOOLKIT_ROOT_DIR};${CMAKE_PREFIX_PATH}" \
      -D CMAKE_INCLUDE_PATH="${VIDEO_CODEC_SDK_DIR}/Interface;${CMAKE_INCLUDE_PATH}" \
      -D CMAKE_LIBRARY_PATH="${VIDEO_CODEC_SDK_DIR}/Lib/linux/stubs;${CMAKE_LIBRARY_PATH}" \
      \
      -D WITH_CUDA=ON \
      -D CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR} \
      -D CUDA_ARCH_BIN="8.6" \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      \
      -D WITH_NVCUVID=ON \
      -D WITH_NVCUVENC=ON \
      \
      -D WITH_FFMPEG=ON \
      -D WITH_GTK=ON \
      -D WITH_OPENGL=ON \
      -D WITH_GSTREAMER=ON \
      \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB_MODULES_ABS_PATH} \
      \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_DOCS=OFF \
      \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE=$(which python3) \
      -D PYTHON3_PACKAGES_PATH_SUFFIX="python$(python3 -c 'import sys; print(str(sys.version_info.major) + "." + str(sys.version_info.minor))')/site-packages" \
      \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      .. # 指向上一级目录 (即 PERSISTENT_WORK_DIR/opencv-VERSION 源码根目录)

# --- 编译 ---
echo ">>> 编译 OpenCV (这将花费很长时间)..."
make -j${NUM_JOBS}

# --- 安装 ---
echo ">>> 安装 OpenCV..."
sudo make install

# --- 后续步骤 (可选但推荐) ---
echo ">>> OpenCV 已安装到 ${INSTALL_DIR}"
echo ">>> 要使用此构建版本, 你可能需要:"
echo "1. 将库路径添加到 LD_LIBRARY_PATH:"
echo "   echo 'export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:\$LD_LIBRARY_PATH' >> ~/.bashrc"
echo "   或者创建一个链接文件 (如果 ${INSTALL_DIR}/lib 不在标准库搜索路径中):"
echo "   echo '${INSTALL_DIR}/lib' | sudo tee /etc/ld.so.conf.d/opencv_custom_tju.conf"
echo "   然后运行 'sudo ldconfig'"
echo ""
echo "2. 如果编译了 Python 绑定, 将 Python 包路径添加到 PYTHONPATH:"
PYTHON_EXECUTABLE=$(which python3)
PYTHON_VERSION_MAJOR=$(${PYTHON_EXECUTABLE} -c 'import sys; print(sys.version_info.major)')
PYTHON_VERSION_MINOR=$(${PYTHON_EXECUTABLE} -c 'import sys; print(sys.version_info.minor)')
EXPECTED_PYTHON_PACKAGES_PATH="${INSTALL_DIR}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages"

if [ -d "${EXPECTED_PYTHON_PACKAGES_PATH}" ]; then
    echo "   echo 'export PYTHONPATH=${EXPECTED_PYTHON_PACKAGES_PATH}:\$PYTHONPATH' >> ~/.bashrc"
    echo "   请重新加载你的 .bashrc (例如: source ~/.bashrc) 或重启终端。"
else
    echo "   Python site-packages 目录未在预期的 ${EXPECTED_PYTHON_PACKAGES_PATH} 找到。"
    echo "   请手动查找 ${INSTALL_DIR}/lib/python*/site-packages 并将其添加到 PYTHONPATH。"
fi
echo ""
echo ">>> 编译和安装完成!"
echo ">>> 请仔细检查 CMake 配置阶段和 make 编译阶段的输出，确保没有错误或重要警告被忽略。"
