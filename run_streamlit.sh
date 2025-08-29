#!/bin/bash
# Usage: ./run_streamlit.sh --start|--kill|--status
# Runs Streamlit server for the Gnosis GUI from the project root using the venv

VENV_DIR=".venv"
APP_PATH="src/gui/app.py"
PID_FILE=".streamlit_server.pid"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/streamlit_server.log"

function start_server() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "Streamlit server already running (PID $(cat $PID_FILE))"
        exit 0
    fi
    if [ ! -d "$VENV_DIR" ]; then
        echo "Virtual environment not found at $VENV_DIR. Please create it first."
        exit 1
    fi
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    nohup streamlit run "$APP_PATH" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Streamlit server started (PID $!)"
}

function kill_server() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 $PID 2>/dev/null; then
            kill $PID
            echo "Streamlit server (PID $PID) killed."
        else
            echo "No running Streamlit server found (stale PID file)."
        fi
        rm -f "$PID_FILE"
    else
        echo "No PID file found. Server may not be running."
    fi
}

function status_server() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 $PID 2>/dev/null; then
            echo "Streamlit server is running (PID $PID)"
        else
            echo "Streamlit server not running, but PID file exists."
        fi
    else
        echo "Streamlit server is not running."
    fi
}

case "$1" in
    --start)
        start_server
        ;;
    --kill)
        kill_server
        ;;
    --status)
        status_server
        ;;
    *)
        echo "Usage: $0 --start|--kill|--status"
        exit 1
        ;;
esac