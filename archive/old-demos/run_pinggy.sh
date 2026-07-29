#!/usr/bin/env bash
# Keep a pinggy tunnel (SSH over 443) to localhost:8000 alive, auto-reconnecting.
cd $NATIONALITY_RUNS/demo
while true; do
  ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
      -R0:localhost:8000 a.pinggy.io
  echo "[$(date)] pinggy dropped, reconnecting in 3s..."
  sleep 3
done
