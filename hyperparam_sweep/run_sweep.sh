#!/usr/bin/env bash
# Auto-restart wrapper for sweep.py.
# Exit code 2 = CUDA crash, resume is safe.
# Exit code 0 = all done.
# Anything else = unexpected error; still try to resume up to MAX_RETRIES.

MAX_RETRIES=20
attempt=0

while [ $attempt -lt $MAX_RETRIES ]; do
    attempt=$((attempt + 1))
    echo "=== Sweep attempt $attempt / $MAX_RETRIES  ($(date)) ==="
    python3 "$(dirname "$0")/sweep.py"
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "=== Sweep completed successfully ==="
        exit 0
    fi
    echo "=== sweep.py exited with code $rc — restarting in 10s ==="
    sleep 10
done

echo "=== Max retries ($MAX_RETRIES) reached ==="
exit 1
