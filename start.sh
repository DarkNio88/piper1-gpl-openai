#!/usr/bin/env bash
set -euo pipefail

# Start Piper HTTP server in a detached screen session
SESSION_NAME="piper_server"
CMD="python3 -m piper.http_server -m it_IT-paola-medium --host 0.0.0.0 --debug"

# If a virtualenv exists in .venv, prefer activating it
#if [ -f ".venv/bin/activate" ]; then
#	CMD=". .venv/bin/activate && ${CMD}"
#fi

if screen -list | grep -q "\.${SESSION_NAME}[[:space:]]"; then
	echo "Screen session '${SESSION_NAME}' already running"
else
	screen -dmS "${SESSION_NAME}" bash -lc "${CMD}"
	echo "Started screen session '${SESSION_NAME}'"
fi