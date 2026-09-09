"""
Firebase Authentication integration.

Client-side (static/js/auth.js) handles the actual sign-in flow (email/
password + Google) using the Firebase Web SDK, then POSTs the resulting
ID token to /api/session-login. This module verifies that token with the
Firebase Admin SDK and stores the verified user in the Flask session â€”
the server never trusts a client-supplied UID without verifying the token
signature first.
"""
import functools
import os

import requests
from flask import jsonify, redirect, session, url_for

_firebase_ready = False
_admin_available = False


def init_firebase_admin():
    """
    Tries to initialise the Firebase Admin SDK using a service-account
    file. If none is configured, verify_id_token() below transparently
    falls back to Firebase's public `accounts:lookup` REST endpoint,
    which still cryptographically validates the ID token server-side â€”
    it just doesn't require a downloaded service-account key.
    """
    global _firebase_ready, _admin_available
    if _firebase_ready:
        return
    cred_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH")
    if cred_path and os.path.exists(cred_path):
        try:
            import firebase_admin
            from firebase_admin import credentials
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            _admin_available = True
        except ValueError:
            _admin_available = True  # already initialized
        except Exception:
            _admin_available = False
    _firebase_ready = True


def _verify_via_admin_sdk(id_token: str) -> dict:
    from firebase_admin import auth as fb_auth
    decoded = fb_auth.verify_id_token(id_token)
    return {
        "uid": decoded["uid"],
        "email": decoded.get("email"),
        "name": decoded.get("name", decoded.get("email", "").split("@")[0]),
        "picture": decoded.get("picture"),
    }


def _verify_via_rest(id_token: str) -> dict:
    api_key = os.environ.get("FIREBASE_API_KEY")
    if not api_key:
        raise RuntimeError("No FIREBASE_API_KEY configured for token verification")
    resp = requests.post(
        f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={api_key}",
        json={"idToken": id_token}, timeout=6,
    )
    resp.raise_for_status()
    users = resp.json().get("users", [])
    if not users:
        raise RuntimeError("Invalid ID token")
    u = users[0]
    return {
        "uid": u["localId"],
        "email": u.get("email"),
        "name": u.get("displayName") or (u.get("email", "").split("@")[0]),
        "picture": u.get("photoUrl"),
    }


def verify_id_token(id_token: str) -> dict:
    init_firebase_admin()
    if _admin_available:
        return _verify_via_admin_sdk(id_token)
    return _verify_via_rest(id_token)


def login_required(view_func):
    @functools.wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("uid"):
            return redirect(url_for("web.login"))
        return view_func(*args, **kwargs)
    return wrapped


def admin_required(view_func):
    @functools.wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("uid"):
            return redirect(url_for("web.login"))
        if not session.get("is_admin"):
            return jsonify({"error": "Admin access required"}), 403
        return view_func(*args, **kwargs)
    return wrapped
