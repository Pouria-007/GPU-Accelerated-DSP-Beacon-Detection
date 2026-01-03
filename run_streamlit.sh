#!/bin/bash
# Run Streamlit app for beacon detection

cd "$(dirname "$0")"
conda run -n cyberone streamlit run app.py

