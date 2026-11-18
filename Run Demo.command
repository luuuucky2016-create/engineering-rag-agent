#!/bin/bash
# Double-click me in Finder. Runs tests + live demo and saves a log for Claude to check.
cd "$(dirname "$0")"
clear
PY=python3
for c in python3.12 python3.11 python3.10 "$HOME/anaconda3/bin/python3" "$HOME/opt/anaconda3/bin/python3" python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
export PY PYTHONUNBUFFERED=1
MEM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 17179869184) / 1073741824 ))
export CHAT_MODEL="${CHAT_MODEL:-qwen2.5:7b}"
echo "RAM: ${MEM_GB} GB -> model $CHAT_MODEL"
bash ./run_demo.sh 2>&1 | tee demo_run_log.txt
echo; echo "Done. You can close this window."
read -r
