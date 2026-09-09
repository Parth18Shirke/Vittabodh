# -*- coding: utf-8 -*-
"""
VittaBodh - entry point.
Run with:  python run.py
Production:  gunicorn -w 2 -k gthread --threads 4 -b 0.0.0.0:5000 run:app
"""
import os
import sys

# Force UTF-8 mode on Windows so Jinja2 reads templates correctly
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()  # picks up .env in the project root

from app import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
