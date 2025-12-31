#!/bin/bash
# Install ngrok if not present (Linux)
if ! command -v ngrok &> /dev/null; then
    echo "Installing ngrok..."
    curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok
fi

# Start Streamlit in background
echo "Starting Streamlit App..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &
PID=$!
echo "Streamlit running (PID $PID). Logs in streamlit.log"

# Start ngrok tunnel
echo "Starting ngrok tunnel..."
ngrok http 8501 > ngrok.log 2>&1 &

sleep 5
# Extract URL
URL=$(grep -o 'https://[^"]*ngrok-free.app' ngrok.log | head -n 1)

if [ -z "$URL" ]; then
    URL=$(grep -o 'https://[^"]*ngrok.io' ngrok.log | head -n 1)
fi

echo "========================================================"
echo "🌍 Dashboard is LIVE at: $URL"
echo "========================================================"
echo "Press Ctrl+C to stop."

wait $PID

