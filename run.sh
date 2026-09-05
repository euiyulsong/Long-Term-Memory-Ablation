python3 chunk_fixed.py \
  --limit 50 \
  --workers 50 \
  --model qwen/qwen3.5-35b-a3b \
  --chunks 2 \
  --methods untyped joint_typed sequential_typed \
  --memory-tokens 300 \
  --qa-tokens 16

python3 locomo_chunk_router.py \
  --num-examples 50 \
  --seed 42 \
  --top-k 5 \
  --max-workers 8 \
  --c6-overlap-stride 3 \
  --qa-model qwen/qwen3.5-35b-a3b \
  --router-model qwen/qwen3.5-35b-a3b \
  --self-router-model qwen/qwen3.5-35b-a3b \
  --output-dir ./results_locomo_final
