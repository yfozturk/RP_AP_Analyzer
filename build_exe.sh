#!/bin/bash
echo "Building ISO9283 Analyzer binary..."
pip install pyinstaller
pyinstaller --onefile --windowed --name "ISO9283_Analyzer" iso9283_analyzer.py
echo "Done! Check the dist/ folder."
