set -e

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/ajjad-matplotlib
PYTHON_BIN="../../../.venv/bin/python"

"$PYTHON_BIN" trainSage.py
echo "training complete for sage gnn"
"$PYTHON_BIN" testsage.py
