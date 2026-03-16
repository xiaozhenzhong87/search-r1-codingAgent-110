#!/bin/bash
# 为K8s concepts corpus构建E5 Flat索引

corpus_file=/ssd1/zz/AI_efficency/RAG/data/k8s-concepts-corpus.jsonl
save_dir=/ssd1/zz/AI_efficency/RAG/data/k8s_index
retriever_name=e5
retriever_model=intfloat/e5-base-v2

echo "===== Building K8s corpus index ====="
echo "Corpus: $corpus_file"
echo "Save dir: $save_dir"
echo "Model: $retriever_model"
echo ""

# 创建输出目录
mkdir -p $save_dir

# 激活retriever环境
source /root/anaconda3/bin/activate /ssd1/zz/envs/retriever

echo "Activated conda environment: retriever"
python --version

# 构建索引 (使用单GPU)
CUDA_VISIBLE_DEVICES=0 python search_r1/search/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 256 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding

echo ""
echo "===== Index building completed! ====="
echo "Index file: $save_dir/${retriever_name}_Flat.index"
