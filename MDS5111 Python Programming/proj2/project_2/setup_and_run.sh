#!/usr/bin/env bash
set -e

if command -v python3.10 >/dev/null 2>&1; then
  python3.10 setup_and_run.py
elif command -v python >/dev/null 2>&1; then
  python setup_and_run.py
else
  echo "Python not found. Please install Python 3.10.x (recommended: 3.10.20)."
  exit 1
fi
