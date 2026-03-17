#!/bin/bash
export GROQ_API_KEY="${GROQ_API_KEY}"
.venv/bin/python -m backend.api &
.venv/bin/python frontend/app.py
