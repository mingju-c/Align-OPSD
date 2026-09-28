#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts/common.sh"
index_file="${SEARCH_INDEX_FILE:-$SEARCH_INDEX_DIR/e5_Flat.index}"
corpus_file="${SEARCH_CORPUS_FILE:-$SEARCH_INDEX_DIR/wiki-18.jsonl}"
for asset in "$index_file" "$corpus_file"; do
    [[ -f "$asset" ]] || { echo "Missing retrieval asset: $asset" >&2; exit 1; }
done
faiss_args=()
if [[ "${FAISS_GPU:-true}" == true ]]; then faiss_args+=(--faiss_gpu); fi
exec python3 "$ALIGNOPSD_ROOT/examples/search/retriever/retrieval_server.py" \
    --index_path "$index_file" --corpus_path "$corpus_file" \
    --topk "${SEARCH_TOPK:-3}" --retriever_name e5 \
    --retriever_model "${RETRIEVER_MODEL_PATH:-intfloat/e5-base-v2}" \
    --port "${SEARCH_PORT:-8000}" "${faiss_args[@]}" "$@"
