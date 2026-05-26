#!/bin/bash
MON_GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
MON_GPU_ID=$(echo $MON_GPU_ID | cut -d',' -f1)

TMP_MEM_FILE=$(mktemp)
echo 0 > "$TMP_MEM_FILE"

(
  max_mem=0
  while true; do
     mem=$(nvidia-smi -i "$MON_GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
     if [ "$mem" -gt "$max_mem" ]; then
        max_mem=$mem
        echo "$max_mem" > "$TMP_MEM_FILE"
     fi
     sleep 0.1
  done
) &
MON_PID=$!

$REAL_PYTHON "$@"
EXIT_CODE=$?

kill $MON_PID 2>/dev/null
wait $MON_PID 2>/dev/null || true

sleep 0.5
REAL_MAX_MEM=$(cat "$TMP_MEM_FILE")
rm -f "$TMP_MEM_FILE"

echo "[VRAM] peak_allocated_mb=$REAL_MAX_MEM" >&2
exit $EXIT_CODE
