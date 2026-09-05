python3 chunk_fixed.py \
  --limit 50 \
  --workers 50 \
  --model qwen/qwen3.5-35b-a3b \
  --chunks 2 \
  --methods untyped joint_typed sequential_typed \
  --memory-tokens 300 \
  --qa-tokens 16
