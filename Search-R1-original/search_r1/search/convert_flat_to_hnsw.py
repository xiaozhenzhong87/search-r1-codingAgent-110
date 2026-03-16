"""Convert an existing FAISS Flat index to HNSW64 index.

Reads vectors directly from the Flat index, so no encoder / GPU needed.
Memory requirement: ~130 GB RAM (Flat index + vectors + HNSW build buffer).
"""

import argparse
import os
import time
import faiss
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat_index", type=str, required=True)
    parser.add_argument("--output_index", type=str, default=None)
    parser.add_argument("--hnsw_m", type=int, default=64,
                        help="Number of neighbors per node in HNSW graph (32/64/128)")
    parser.add_argument("--ef_construction", type=int, default=200,
                        help="efConstruction: higher = better quality, slower build")
    args = parser.parse_args()

    if args.output_index is None:
        dirname = os.path.dirname(args.flat_index)
        args.output_index = os.path.join(dirname, f"e5_HNSW{args.hnsw_m}.index")

    print(f"[1/4] Loading Flat index: {args.flat_index}")
    t0 = time.time()
    flat_index = faiss.read_index(args.flat_index)
    ntotal = flat_index.ntotal
    d = flat_index.d
    print(f"       Loaded {ntotal:,} vectors, dim={d}  ({time.time()-t0:.1f}s)")

    print(f"[2/4] Extracting vectors from Flat index ...")
    t1 = time.time()
    xb = faiss.rev_swig_ptr(flat_index.get_xb(), ntotal * d).reshape(ntotal, d).copy()
    del flat_index
    print(f"       Shape: {xb.shape}, dtype: {xb.dtype}  ({time.time()-t1:.1f}s)")

    print(f"[3/4] Building HNSW{args.hnsw_m} index (efConstruction={args.ef_construction}) ...")
    t2 = time.time()
    hnsw_index = faiss.IndexHNSWFlat(d, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    hnsw_index.hnsw.efConstruction = args.ef_construction
    hnsw_index.hnsw.efSearch = 128
    hnsw_index.add(xb)
    del xb
    print(f"       Added {hnsw_index.ntotal:,} vectors  ({time.time()-t2:.1f}s)")

    print(f"[4/4] Saving to {args.output_index} ...")
    t3 = time.time()
    faiss.write_index(hnsw_index, args.output_index)
    size_gb = os.path.getsize(args.output_index) / 1e9
    print(f"       Done! File size: {size_gb:.1f} GB  ({time.time()-t3:.1f}s)")
    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
