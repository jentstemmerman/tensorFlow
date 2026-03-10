#!/bin/bash
set -e

echo "=== LogFileAnalyzer Container Starting ==="

# Ensure data directory exists
mkdir -p /app/data

# Start Gunicorn (Flask + Socket.IO) in background
cd /app/Server
# Ensure LogFileAnalyzer is on the Python path (server imports from it via sys.path.append)
export PYTHONPATH="/app/LogFileAnalyzer:${PYTHONPATH:-}"
echo "Starting Gunicorn on 127.0.0.1:5000 ..."
gunicorn \
    --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker \
    --workers 1 \
    --bind 127.0.0.1:5000 \
    --timeout 600 \
    --access-logfile - \
    --error-logfile - \
    WebServer:WebServer &

GUNICORN_PID=$!

# Give Gunicorn a moment to start
sleep 2

# Start Nginx in foreground
echo "Starting Nginx on port 80 ..."
nginx -g "daemon off;" &
NGINX_PID=$!

# Graceful shutdown handler
shutdown() {
    echo "Shutting down ..."
    kill -SIGTERM "$NGINX_PID" 2>/dev/null || true
    kill -SIGTERM "$GUNICORN_PID" 2>/dev/null || true
    wait "$NGINX_PID" "$GUNICORN_PID" 2>/dev/null || true
    echo "Shutdown complete."
    exit 0
}

trap shutdown SIGTERM SIGINT SIGQUIT

echo "=== Container ready ==="

# Wait for either process to exit
wait -n "$GUNICORN_PID" "$NGINX_PID" 2>/dev/null || true

# If one exits, stop the other
shutdown
