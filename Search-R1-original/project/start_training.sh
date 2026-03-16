#!/bin/bash
#
# K8s RAG Search-R1 训练快速启动脚本
# 自动检查依赖并启动训练
#
# 用法:
#   ./start_training.sh           # 交互模式（有警告时会询问）
#   ./start_training.sh --yes     # 自动模式（跳过确认）
#   nohup bash start_training.sh --yes &   # 后台运行
#

set -euo pipefail

# 解析命令行参数
AUTO_YES=false
if [ $# -gt 0 ] && [ "$1" = "--yes" ]; then
    AUTO_YES=true
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "  K8s RAG Search-R1 训练准备检查"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SUCCESS=0
WARNINGS=0

# 检查 1: 数据文件
echo -n "检查训练数据 (Parquet)... "
TRAIN_DATA="/ssd1/zz/AI_efficency/RAG/data/k8s_qa_data_2k/qa_2000_searchr1_format.parquet"
if [ -f "$TRAIN_DATA" ]; then
    echo -e "${GREEN}✓${NC}"
    FILE_SIZE=$(du -h "$TRAIN_DATA" | cut -f1)
    echo "  → 文件大小: $FILE_SIZE"
else
    echo -e "${RED}✗${NC}"
    echo -e "${YELLOW}  → 需要先转换数据格式${NC}"
    
    JSON_DATA="/ssd1/zz/AI_efficency/RAG/data/k8s_qa_data_2k/qa_2000_searchr1_format.json"
    if [ -f "$JSON_DATA" ]; then
        echo ""
        echo "正在转换 JSON → Parquet..."
        python project/convert_json_to_parquet.py "$JSON_DATA" "$TRAIN_DATA"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ 转换成功${NC}"
        else
            echo -e "${RED}✗ 转换失败${NC}"
            exit 1
        fi
    else
        echo -e "${RED}  → 找不到源 JSON 文件: $JSON_DATA${NC}"
        exit 1
    fi
fi
echo ""

# 检查 2: 模型路径
echo -n "检查基础模型... "
BASE_MODEL="/ssd1/zz/models/Qwen/Qwen2.5-7B-Instruct"
if [ -d "$BASE_MODEL" ]; then
    echo -e "${GREEN}✓${NC}"
    echo "  → 路径: $BASE_MODEL"
else
    echo -e "${RED}✗${NC}"
    echo -e "${RED}  → 找不到模型: $BASE_MODEL${NC}"
    exit 1
fi
echo ""

# 检查 3: GPU 可用性
echo -n "检查 GPU... "
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "${GREEN}✓${NC}"
    echo "  → 可用 GPU: $GPU_COUNT 块"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read line; do
        echo "    - $line"
    done
    
    if [ "$GPU_COUNT" -lt 8 ]; then
        echo -e "${YELLOW}  ⚠ 警告: 可用 GPU 少于 8 块，训练可能较慢${NC}"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${RED}✗${NC}"
    echo -e "${RED}  → nvidia-smi 不可用${NC}"
    exit 1
fi
echo ""

# 检查 4: 检索服务
echo -n "检查检索服务... "
RETRIEVER_URL="http://127.0.0.1:8001/retrieve"
if curl -s --connect-timeout 3 "$RETRIEVER_URL" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
    echo "  → 服务地址: $RETRIEVER_URL"
else
    echo -e "${YELLOW}✗${NC}"
    echo -e "${YELLOW}  → 无法连接到: $RETRIEVER_URL${NC}"
    echo -e "${YELLOW}  ⚠ 警告: 检索服务未启动，训练时会失败！${NC}"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# 检查 5: Python 环境
echo -n "检查 Python 环境... "
if [ -f "/ssd1/zz/envs/searchr1/bin/python" ]; then
    echo -e "${GREEN}✓${NC}"
    PYTHON_VERSION=$(/ssd1/zz/envs/searchr1/bin/python --version 2>&1)
    echo "  → $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC}"
    echo -e "${RED}  → 找不到 searchr1 环境${NC}"
    exit 1
fi
echo ""

# 检查 6: WandB 配置
echo -n "检查 WandB... "
if /ssd1/zz/envs/searchr1/bin/python -c "import wandb" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
    WANDB_API_KEY=$(grep -r "api_key" ~/.netrc 2>/dev/null | head -1 || echo "")
    if [ -n "$WANDB_API_KEY" ]; then
        echo "  → 已配置 API key"
    else
        echo -e "${YELLOW}  ⚠ 未检测到 API key，可能需要登录${NC}"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${YELLOW}✗${NC}"
    echo -e "${YELLOW}  → WandB 未安装，日志可能无法上传${NC}"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# 检查 7: 磁盘空间
echo -n "检查磁盘空间... "
DISK_AVAILABLE=$(df -BG /ssd1 | tail -1 | awk '{print $4}' | sed 's/G//')
echo -e "${GREEN}✓${NC}"
echo "  → 可用空间: ${DISK_AVAILABLE}GB"
if [ "$DISK_AVAILABLE" -lt 100 ]; then
    echo -e "${YELLOW}  ⚠ 警告: 磁盘空间不足 100GB，检查点可能无法保存${NC}"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# 汇总结果
echo "=========================================="
if [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}检查完成: 发现 $WARNINGS 个警告（自动继续）${NC}"
else
    echo -e "${GREEN}检查完成: 所有依赖就绪 ✓${NC}"
fi
echo "=========================================="
echo ""

# 启动训练（移除所有交互逻辑）
echo ""
echo "=========================================="
echo "  启动训练"
echo "=========================================="
echo ""

exec ./project/train_k8s_rag_search_r1.sh
