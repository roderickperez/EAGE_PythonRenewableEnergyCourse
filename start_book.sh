#!/bin/bash
# Start the EAGE Jupyter Book server
# Run from: /home/roderickperez/DataScienceProjects/EAGE_Python_course/

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_JUPYTER="$SCRIPT_DIR/.venv/bin/jupyter"
BOOK_DIR="$SCRIPT_DIR/EAGE_PythonRenewableEnergyCourse"

cd "$BOOK_DIR" && "$VENV_JUPYTER" book start
