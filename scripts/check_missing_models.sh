#!/bin/bash

echo "======================================================================="
echo " Checking OpenCV Build Prerequisites for Ubuntu 22.04                  "
echo "======================================================================="
echo " Target: OpenCV build with CUDA support (requires manual verification) "
echo "         and standard modules (incl. Python bindings).                 "
echo ""

# --- List of packages to check ---
# Essential build tools
packages_build="build-essential cmake pkg-config"

# Image I/O libraries
packages_img="libjpeg-dev libpng-dev libtiff-dev libwebp-dev libopenjp2-7-dev"

# Video I/O libraries (Using FFmpeg)
packages_vid="libavcodec-dev libavformat-dev libswscale-dev libswresample-dev" # Corrected package

# GUI Library (Using GTK+3) - Change if you prefer Qt5 (e.g., qtbase5-dev)
packages_gui="libgtk-3-dev"

# Parallelism & Optimization (Recommended)
packages_perf="libtbb-dev libeigen3-dev"

# Python 3 Bindings support
packages_py="python3-dev python3-numpy"

# Combine all package lists
all_packages="$packages_build $packages_img $packages_vid $packages_gui $packages_perf $packages_py"
# --- End of package list ---


missing_packages=()
all_os_packages_installed=true

echo ">>> Checking required APT packages..."

# Loop through each package and check if it's installed
for pkg in $all_packages; do
    # dpkg-query suppresses errors, check exit status
    dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"
    if [ $? -ne 0 ]; then
        missing_packages+=("$pkg")
        all_os_packages_installed=false
        # echo "    - $pkg: MISSING" # Uncomment for verbose output
    # else
        # echo "    - $pkg: Installed" # Uncomment for verbose output
    fi
done

echo ""
echo ">>> APT Package Check Summary:"

if $all_os_packages_installed; then
    echo "    All checked OS prerequisite packages seem to be installed."
    echo "    Remember to manually verify CUDA Toolkit, cuDNN, and Video Codec SDK!"
else
    echo "    The following OS prerequisite packages appear to be MISSING:"
    printf "      %s\n" "${missing_packages[@]}"
    echo ""
    echo "    You can try installing the missing packages using:"
    # Construct the install command suggestion
    install_command="sudo apt update && sudo apt install -y"
    for missing in "${missing_packages[@]}"; do
        install_command+=" $missing"
    done
    echo "    $install_command"
    echo ""
    echo "    Alternatively, to install ALL listed recommended packages (apt skips installed ones):"
    echo "    sudo apt update && sudo apt install -y $all_packages"
fi

echo ""
echo "======================================================================="
echo " IMPORTANT MANUAL VERIFICATION REQUIRED for NVIDIA Components:         "
echo "======================================================================="
echo " This script CANNOT verify manually installed NVIDIA software."
echo " Please ensure the following are correctly installed and configured:"
echo ""
echo " 1. NVIDIA Driver:"
echo "    - Check with: nvidia-smi"
echo "    - (You have Driver 550.163.01 - OK)"
echo ""
echo " 2. CUDA Toolkit (e.g., 12.4):"
echo "    - Check if installation path exists (e.g., /usr/local/cuda-12.4)"
echo "    - Check if nvcc command works: nvcc --version"
echo "    - Ensure CUDA environment variables (PATH, LD_LIBRARY_PATH) are set correctly (often in ~/.bashrc)."
echo ""
echo " 3. cuDNN (if using OPENCV_DNN_CUDA=ON):"
echo "    - Check if downloaded from NVIDIA Dev Zone (requires registration)."
echo "    - Check if header/library files are copied into your CUDA Toolkit installation directory."
echo "    - Verify compatibility with CUDA 12.4."
echo ""
echo " 4. NVIDIA Video Codec SDK (if using WITH_NVCUVID=ON):"
echo "    - Check if downloaded from NVIDIA Dev Zone."
echo "    - Check if extracted to a known location (you might need to point CMake to it)."
echo ""
echo "======================================================================="

# End of script
