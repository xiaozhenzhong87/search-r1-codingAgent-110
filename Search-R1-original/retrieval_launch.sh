
file_path=/ssd1/zz/AI_efficency/RAG/Search-R1/data/retrieval
# index_file=$file_path/e5_Flat.index       # old: Flat + GPU
index_file=$file_path/e5_HNSW64.index        # new: HNSW64 + CPU (no GPU needed)
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

export HF_HOME=/ssd1/zz/.cache/huggingface

# HNSW index runs on CPU; encoder still uses GPU (~0.5GB) for fast query encoding
# python scripts/simple_retrieval_server.py --index_path $index_file \
python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
                                            # --port 8000
