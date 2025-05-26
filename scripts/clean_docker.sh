#!/bin/bash

# 设置颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Docker 清理脚本 ===${NC}"
echo ""

# --- 第一步：清理所有已停止的容器 ---
echo -e "${GREEN}1. 正在清理所有已停止的容器...${NC}"
# docker container prune 命令用于移除所有已停止的容器
# -f 或 --force 标志用于跳过确认提示
# 如果你想手动确认每个操作，请删除 -f 标志
sudo docker container prune -f

if [ $? -eq 0 ]; then
    echo -e "${GREEN}   所有已停止的容器已清理完毕。${NC}"
else
    echo -e "${RED}   清理已停止容器时发生错误。${NC}"
fi
echo ""

# --- 第二步：清理所有悬空（dangling）镜像 ---
echo -e "${GREEN}2. 正在清理所有悬空（dangling）镜像...${NC}"
# docker image prune 命令用于移除所有悬空镜像
# -f 或 --force 标志用于跳过确认提示
# 如果你想手动确认每个操作，请删除 -f 标志
sudo docker image prune -f

if [ $? -eq 0 ]; then
    echo -e "${GREEN}   所有悬空镜像已清理完毕。${NC}"
else
    echo -e "${RED}   清理悬空镜像时发生错误。${NC}"
fi
echo ""

# --- 第三步：清理所有未被使用的镜像（可选，更激进）---
# 慎用！此操作将删除所有未被任何容器使用的镜像，包括有标签的镜像。
# 默认情况下，此部分被注释掉。如果你需要执行此操作，请取消注释。
#
# echo -e "${YELLOW}3. (可选) 正在清理所有未被使用的镜像 (包括有标签的)...${NC}"
# read -p "此操作将删除所有未被容器使用的镜像。确认继续？(y/N): " confirm_all_images
# if [[ "$confirm_all_images" =~ ^[Yy]$ ]]; then
#     # --all 或 -a 标志用于删除所有未使用的镜像，而不仅仅是悬空镜像
#     sudo docker image prune --all -f
#     if [ $? -eq 0 ]; then
#         echo -e "${GREEN}   所有未被使用的镜像已清理完毕。${NC}"
#     else
#         echo -e "${RED}   清理所有未使用的镜像时发生错误。${NC}"
#     fi
# else
#     echo -e "${YELLOW}   已跳过清理所有未被使用的镜像。${NC}"
# fi
# echo ""

echo -e "${YELLOW}=== Docker 清理完成 ===${NC}"
echo "你可以运行 'docker ps -a' 和 'docker images' 来验证清理结果。"
