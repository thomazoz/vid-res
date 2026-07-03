#!/bin/zsh
# Double-click to launch the Video Research Hub.
cd "$(dirname "$0")"

PORT=8501

# If the hub is already running, just open the browser tab.
if curl -s "http://localhost:$PORT/_stcore/health" 2>/dev/null | grep -q ok; then
    echo "Hub already running — opening browser."
    open "http://localhost:$PORT"
    exit 0
fi

echo "Starting Video Research Hub… (close this window / Ctrl-C to stop it)"
exec /opt/anaconda3/bin/streamlit run hub.py --server.port $PORT
