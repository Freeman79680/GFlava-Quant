#!/usr/bin/env bash
# GFlava-Quant launcher for Linux and macOS (the counterpart of start_server.bat).
#
# Usage:
#   ./start_server.sh                          use the first Python it finds
#   GFLAVA_PYTHON=/path/to/venv/bin/python ./start_server.sh
#
# Use the Python of the environment where ComfyUI and convert_to_quant live, so
# the server finds ctq and the packages. Python lookup order:
#   1. $GFLAVA_PYTHON
#   2. venv/bin/python or .venv/bin/python next to this script
#   3. python3 (or python) on your PATH

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="$SCRIPT_DIR/quant_server.py"
URL="http://127.0.0.1:8877"

find_python() {
    if [ -n "${GFLAVA_PYTHON:-}" ]; then
        echo "$GFLAVA_PYTHON"
        return
    fi
    local candidate
    for candidate in "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/.venv/bin/python"; do
        if [ -x "$candidate" ]; then
            echo "$candidate"
            return
        fi
    done
    command -v python3 || command -v python || true
}

PYTHON="$(find_python)"
if [ -z "$PYTHON" ] || ! "$PYTHON" --version >/dev/null 2>&1; then
    echo "[ERROR] No working Python found."
    echo "Set GFLAVA_PYTHON to the python of your ComfyUI environment, for example:"
    echo "  GFLAVA_PYTHON=\$HOME/ComfyUI/venv/bin/python ./start_server.sh"
    exit 1
fi
echo "Using Python: $PYTHON"

if [ ! -f "$SERVER_SCRIPT" ]; then
    echo "[ERROR] quant_server.py was not found next to this script ($SCRIPT_DIR)."
    exit 1
fi
if [ ! -f "$SCRIPT_DIR/templates/index.html" ] || [ ! -f "$SCRIPT_DIR/static/app.js" ]; then
    echo "[ERROR] The folders \"templates\" and \"static\" were not found next to quant_server.py:"
    echo "  $SCRIPT_DIR"
    echo "GFlava-Quant needs them since version 1.3 -- please copy the complete folder."
    exit 1
fi

# Offer to install what is missing, with a short explanation of what it is for.
ask_install() {
    local module="$1" package="$2" purpose="$3" answer
    echo
    echo "'$package' is not installed. $purpose"
    printf "Install it now with pip? [y/N] "
    read -r answer
    case "$answer" in
        y|Y|yes|YES)
            if ! "$PYTHON" -m pip install "$package"; then
                echo
                echo "[ERROR] pip could not install '$package'."
                echo "If the message above says \"externally-managed-environment\", this Python belongs to"
                echo "your system. Use a virtual environment instead (python3 -m venv venv) and start with"
                echo "  GFLAVA_PYTHON=\$PWD/venv/bin/python ./start_server.sh"
                exit 1
            fi
            ;;
        *)
            echo "Skipped. GFlava-Quant may not work without '$package'."
            ;;
    esac
}

if ! "$PYTHON" -c "import flask" >/dev/null 2>&1; then
    ask_install flask flask "It runs the local web page you are about to open."
fi
if ! "$PYTHON" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('convert_to_quant') else 1)" >/dev/null 2>&1; then
    ask_install convert_to_quant convert_to_quant "It is the program that does the actual quantization."
fi

echo "Starting GFlava-Quant at $URL  (Ctrl+C stops the server)"
# --open-browser: the server opens the page itself as soon as it accepts connections.
exec "$PYTHON" "$SERVER_SCRIPT" --open-browser
