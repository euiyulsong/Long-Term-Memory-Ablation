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

python3 locomo_pointwise_selfrouter.py \
  --num-examples 50 \
  --seed 42 \
  --top-k 5 \
  --max-workers 10 \
  --c6-overlap-stride 3 \
  --model qwen/qwen3.5-35b-a3b \
  --output-dir ./results_locomo_pointwise_selfrouter

python3 locomo_no_selfrouter.py \
  --num-examples 50 \
  --seed 42 \
  --top-k 5 \
  --max-workers 8 \
  --c6-overlap-stride 3 \
  --model qwen/qwen3.5-35b-a3b \
  --output-dir ./results_locomo_no_selfrouter

python3 locomo_c6_summary_concat.py \
  --num-examples 50 \
  --seed 42 \
  --top-k 5 \
  --c6-overlap-stride 3 \
  --model qwen/qwen3.5-35b-a3b \
  --output-dir ./results_locomo_c6_summary_concat

python3 locomo_task_router_final.py \
  --num-examples 50 \
  --seed 42 \
  --top-k 5 \
  --c6-overlap-stride 3 \
  --router-model qwen/qwen3.5-35b-a3b \
  --qa-model qwen/qwen3.5-35b-a3b \
  --output-dir ./results_locomo_task_router_final
