#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}
log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}
log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}
log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 command not found. Please install it."
        return 1
    fi
    return 0
}

check_path_exists() {
    local path_to_check=$1
    local type_desc=$2
    if [ ! -e "$path_to_check" ]; then
        log_error "$type_desc '$path_to_check' does not exist."
        return 1
    fi
    if [[ "$type_desc" == "directory" ]] && [ ! -d "$path_to_check" ]; then
        log_error "'$path_to_check' is not a directory."
        return 1
    fi
    if [[ "$type_desc" == "file" ]] && [ ! -f "$path_to_check" ]; then
        log_error "'$path_to_check' is not a file."
        return 1
    fi
    return 0
}

check_opencv_path() {
    local opencv_path="/opt/opencv"
    log_info "Checking OpenCV path: $opencv_path"
    check_path_exists "$opencv_path" "OpenCV base directory" || return 1
    check_path_exists "$opencv_path/include/opencv4" "OpenCV include directory" || return 1
    check_path_exists "$opencv_path/lib" "OpenCV lib directory" || return 1
    log_success "OpenCV path check passed for: $opencv_path"
}

check_cuda() {
    log_info "Checking CUDA installation..."
    if command -v nvcc &> /dev/null; then
        :
    elif [ -f "/usr/local/cuda-12.4/targets/x86_64-linux/bin/nvcc" ]; then
        log_info "Found CUDA at /usr/local/cuda-12.4/targets/x86_64-linux. Adding its bin to PATH for this session."
        export PATH="/usr/local/cuda-12.4/targets/x86_64-linux/bin:$PATH"
    elif [ -f "/usr/local/cuda-12.4/bin/nvcc" ]; then
        log_info "Found CUDA at /usr/local/cuda-12.4. Adding its bin to PATH for this session."
        export PATH="/usr/local/cuda-12.4/bin:$PATH"
    else
        log_error "CUDA (nvcc compiler) not found in PATH or at expected /usr/local/cuda-12.4 locations."
        return 1
    fi
    local nvcc_path=$(command -v nvcc)
    local cuda_version=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    log_success "Using CUDA compiler (nvcc): $nvcc_path (Version: $cuda_version)"
}

check_ffmpeg_dev_libs() {
    log_info "Checking for FFmpeg development libraries (libavformat, libavcodec, libavutil, libswscale) via pkg-config..."
    check_command pkg-config || return 1
    
    local ffmpeg_modules="libavformat libavcodec libavutil libswscale"
    if pkg-config --exists $ffmpeg_modules; then
        log_success "FFmpeg development libraries ($ffmpeg_modules) found via pkg-config."
    else
        log_error "pkg-config could not find all required FFmpeg modules: $ffmpeg_modules."
        log_error "Please ensure FFmpeg development packages (e.g., libavformat-dev, libswscale-dev) are installed and pkg-config can find them."
        return 1
    fi
}

check_gpu() {
    check_command nvidia-smi || return 1
    log_info "Listing NVIDIA GPUs:"
    nvidia-smi -L
    log_success "NVIDIA GPU check passed (nvidia-smi found)."
}

setup_environment() {
    local opencv_path_to_setup=$1
    log_info "Setting up environment variables..."

    local current_pkg_config_path="${PKG_CONFIG_PATH:-}"
    PKG_CONFIG_PATH=""

    local opencv_pkgconfig_path="$opencv_path_to_setup/lib/pkgconfig"
    if [ -d "$opencv_pkgconfig_path" ]; then
        PKG_CONFIG_PATH="$opencv_pkgconfig_path${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
        log_info "Prepended to PKG_CONFIG_PATH: $opencv_pkgconfig_path"
    else
        log_warning "OpenCV pkgconfig directory not found: $opencv_pkgconfig_path"
    fi

    local usr_local_pkgconfig_path="/usr/local/lib/pkgconfig"
    if [ -d "$usr_local_pkgconfig_path" ]; then
        PKG_CONFIG_PATH="$usr_local_pkgconfig_path${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH"
        log_info "Prepended to PKG_CONFIG_PATH: $usr_local_pkgconfig_path"
    fi
    
    if [ -n "$current_pkg_config_path" ]; then
         PKG_CONFIG_PATH="${PKG_CONFIG_PATH:+$PKG_CONFIG_PATH:}$current_pkg_config_path"
    fi
    export PKG_CONFIG_PATH
    log_info "Final PKG_CONFIG_PATH=$PKG_CONFIG_PATH"

    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    if [ -d "$opencv_path_to_setup/lib" ]; then
        export LD_LIBRARY_PATH="$opencv_path_to_setup/lib${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
    fi

    local confirmed_cuda_lib_path="/usr/local/cuda-12.4/targets/x86_64-linux/lib"
    if [ -d "$confirmed_cuda_lib_path" ]; then
        export LD_LIBRARY_PATH="$confirmed_cuda_lib_path${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
    else
        log_warning "Confirmed CUDA lib path $confirmed_cuda_lib_path NOT FOUND for LD_LIBRARY_PATH."
    fi
    
    local usr_local_lib_path="/usr/local/lib" 
    if [ -d "$usr_local_lib_path" ]; then 
        export LD_LIBRARY_PATH="$usr_local_lib_path${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
    fi
    log_info "Updated LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
}

compile() {
    local output_name="$1"
    local source_file="$2"
    
    log_info "Starting compilation of $source_file into $output_name..."

    local opencv_cflags=""
    local opencv_libs=""
    local ffmpeg_cflags=""
    local ffmpeg_libs=""
    local cuda_cflags=""
    local cuda_ldflags="" 
    local cuda_link_libs="-lcudart"

    if pkg-config --exists opencv4; then
        log_info "Using pkg-config for OpenCV flags."
        opencv_cflags=$(pkg-config --cflags opencv4)
        opencv_libs=$(pkg-config --libs opencv4)
    else
        log_error "pkg-config could not find opencv4. Please ensure opencv4.pc is in PKG_CONFIG_PATH."
        log_info "Falling back to manual OpenCV flags for /opt/opencv."
        opencv_cflags="-I/opt/opencv/include/opencv4"
        opencv_libs="-L/opt/opencv/lib -lopencv_cudacodec -lopencv_videoio -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_cudaarithm -lopencv_cudawarping -lopencv_dnn_objdetect -lopencv_imgcodecs -lopencv_dnn_superres -lopencv_dnn -lopencv_imgproc -lopencv_core -lopencv_cudev"
    fi

    local ffmpeg_modules_for_compile="libavformat libavcodec libavutil libswscale"
    if pkg-config --exists $ffmpeg_modules_for_compile; then
        log_info "Using pkg-config for FFmpeg flags ($ffmpeg_modules_for_compile)."
        ffmpeg_cflags=$(pkg-config --cflags $ffmpeg_modules_for_compile)
        ffmpeg_libs=$(pkg-config --libs $ffmpeg_modules_for_compile)
    else
        log_error "pkg-config could not find all FFmpeg modules ($ffmpeg_modules_for_compile)."
        log_info "Falling back to manual FFmpeg flags (ensure -lswscale is included if needed by your code)."
        ffmpeg_cflags=""
        ffmpeg_libs="-lavformat -lavcodec -lavutil -lswscale -lswresample -lz -lm -pthread" 
    fi
    
    local confirmed_cuda_base_path="/usr/local/cuda-12.4"
    local confirmed_cuda_include_path="$confirmed_cuda_base_path/targets/x86_64-linux/include"
    local confirmed_cuda_lib_path="$confirmed_cuda_base_path/targets/x86_64-linux/lib"

    if check_path_exists "$confirmed_cuda_include_path" "CUDA include directory"; then
        cuda_cflags="-I$confirmed_cuda_include_path"
    else
        return 1
    fi

    if check_path_exists "$confirmed_cuda_lib_path" "CUDA library directory"; then
        cuda_ldflags="-L$confirmed_cuda_lib_path"
    else
        log_error "FATAL: Confirmed CUDA library path $confirmed_cuda_lib_path NOT FOUND."
        return 1
    fi
    
    local opencv_cuda_def="-DHAVE_CUDA=1"

    log_info "Compiler CFLAGS: $opencv_cflags $ffmpeg_cflags $cuda_cflags $opencv_cuda_def"
    log_info "Compiler LDFLAGS (for CUDA): $cuda_ldflags"
    log_info "Compiler LIBS: $opencv_libs $ffmpeg_libs $cuda_link_libs"

    g++ -std=c++17 -o "$output_name" "$source_file" \
        $opencv_cflags \
        $ffmpeg_cflags \
        $cuda_cflags \
        "$opencv_cuda_def" \
        $cuda_ldflags \
        -Wl,-rpath=/usr/local/lib \
        -Wl,-rpath=/opt/opencv/lib \
        -Wl,-rpath=$confirmed_cuda_lib_path \
        $opencv_libs \
        $ffmpeg_libs \
        $cuda_link_libs \
        -pthread \
        -g

    if [ $? -ne 0 ]; then
        log_error "Compilation failed."
        return 1
    fi

    log_success "Compilation successful. Output: $output_name"
}

main() {
    if [ -z "$1" ]; then
        log_error "Usage: $0 <source_file.cpp> [output_name]"
        log_info "Example: $0 my_program.cpp my_app"
        exit 1
    fi

    local source_file="$1"
    local output_name="${2:-app_$(basename "$source_file" .cpp)}"

    log_info "===== Starting Build Script for $source_file ====="

    check_command g++ || exit 1
    check_command pkg-config || log_warning "pkg-config not found. Dependency flag generation will rely on manual fallbacks."
    
    check_path_exists "$source_file" "Source code file" || exit 1
    
    local opencv_install_path="/opt/opencv"
    setup_environment "$opencv_install_path"

    check_opencv_path "$opencv_install_path" || exit 1
    check_cuda || exit 1
    check_ffmpeg_dev_libs || exit 1
    check_gpu || log_warning "GPU check indicated issues or no GPU found. CUDA-dependent parts may fail at runtime."
    

    if ! compile "$output_name" "$source_file"; then
        exit 1
    fi

    log_success "===== Build Script Finished Successfully ====="
    log_info "To run: ./$output_name <video_path> <interval>"
}

main "$@"