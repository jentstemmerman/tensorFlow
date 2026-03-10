"""JWT authentication middleware for Flask.

Provides:
    - ``create_access_token(user_id, username, role)`` → signed JWT (short-lived)
    - ``create_refresh_token(user_id)`` → opaque hex token stored in UserDB
    - ``@require_auth`` decorator that validates the ``Authorization: Bearer <token>``
      header and injects ``g.user_id``, ``g.username``, ``g.user_role``.

Environment variables:
    JWT_SECRET       – HMAC signing key (REQUIRED in production)
    ACCESS_TOKEN_TTL – Access token lifetime in minutes (default 15)
    REFRESH_TOKEN_TTL – Refresh token lifetime in days (default 7)
"""

import os
import datetime
import secrets
from functools import wraps

import jwt  # PyJWT
from flask import request, g, jsonify

# ---- Configuration ----

_JWT_SECRET = os.environ.get("JWT_SECRET", "CHANGE-ME-IN-PRODUCTION-" + secrets.token_hex(16))
_ACCESS_TTL_MINUTES = int(os.environ.get("ACCESS_TOKEN_TTL", "15"))
_REFRESH_TTL_DAYS = int(os.environ.get("REFRESH_TOKEN_TTL", "7"))
_ALGORITHM = "HS256"


# ---- Token creation ----

def create_access_token(user_id: int, username: str, role: str = "user") -> str:
    """Create a short-lived access JWT."""
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "iat": now,
        "exp": now + datetime.timedelta(minutes=_ACCESS_TTL_MINUTES),
        "type": "access",
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_ALGORITHM)


def create_refresh_token(user_id: int, user_db) -> tuple[str, datetime.datetime]:
    """Create a refresh token, store it in the DB, and return (token, expires_at)."""
    token = secrets.token_hex(48)
    expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=_REFRESH_TTL_DAYS)
    user_db.store_refresh_token(user_id, token, expires_at)
    return token, expires_at


# ---- Token validation ----

def decode_access_token(token: str) -> dict | None:
    """Decode and validate an access token. Returns payload dict or None."""
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_ALGORITHM])
        if payload.get("type") != "access":
            return None
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ---- Flask decorator ----

def require_auth(f):
    """Decorator that enforces JWT authentication on a Flask route.

    On success, sets:
        g.user_id   – int
        g.username  – str
        g.user_role – str
    On failure, returns 401 JSON response.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401

        token = auth_header[7:]  # strip "Bearer "
        payload = decode_access_token(token)
        if payload is None:
            return jsonify({"error": "Invalid or expired token"}), 401

        g.user_id = int(payload["sub"])
        g.username = payload.get("username", "")
        g.user_role = payload.get("role", "user")
        return f(*args, **kwargs)
    return decorated


def validate_socketio_token(auth_data: dict | None) -> dict | None:
    """Validate a JWT passed in Socket.IO's ``auth`` parameter.

    Expected format: ``{ "token": "<access_jwt>" }``

    Returns the decoded payload dict on success, None on failure.
    """
    if not auth_data or not isinstance(auth_data, dict):
        return None
    token = auth_data.get("token")
    if not token:
        return None
    return decode_access_token(token)
