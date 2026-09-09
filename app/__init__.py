import os
from datetime import timedelta

from flask import Flask


def create_app():
    app = Flask(__name__, static_folder="../static", template_folder="../templates")

    # Ensure Jinja2 reads templates as UTF-8 regardless of Windows locale
    app.jinja_env.keep_trailing_newline = True
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    # -- Core config --------------------------------------------------------
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
    app.config["JSON_SORT_KEYS"] = False
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=8)

    # -- Session cookie security --------------------------------------------
    is_prod = os.environ.get("FLASK_ENV", "development") == "production"
    app.config["SESSION_COOKIE_SECURE"] = is_prod   # HTTPS-only in prod
    app.config["SESSION_COOKIE_HTTPONLY"] = True     # JS cannot read cookie
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"   # CSRF mitigation

    # -- Structured JSON logging --------------------------------------------
    from .logging_config import setup_logging
    setup_logging(app)

    # -- Rate limiter (app-less init, bound here) ---------------------------
    from .limiter import limiter
    limiter.init_app(app)

    # -- Security response headers ------------------------------------------
    @app.after_request
    def add_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        # CSP: permissive enough for Firebase Web SDK v10 + Chart.js CDN
        response.headers.setdefault("Content-Security-Policy", (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' "
            "https://www.gstatic.com https://apis.google.com "
            "https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' https://*.googleapis.com "
            "https://*.firebaseio.com "
            "https://identitytoolkit.googleapis.com; "
            "frame-src https://accounts.google.com;"
        ))
        return response

    # â”€â”€ Jinja2 custom filters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @app.template_filter("format_num")
    def format_num(value):
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return value

    # â”€â”€ Blueprints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    from .routes import api_bp, web_bp
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    # -- Pre-load the model at startup --------------------------------------
    from .predict import load_artifacts
    with app.app_context():
        try:
            load_artifacts()
        except Exception as exc:
            app.logger.warning(f"Model artifacts not loaded at startup: {exc}")

    return app
