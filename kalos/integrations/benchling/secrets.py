"""Encryption at rest for a tenant's Benchling app secret.

The key is derived (HKDF-SHA256) from `KALOS_SECRET_KEY`, which only the
operator sets (environment or secret manager; never a file in the repo). Each value
is sealed with AES-256-GCM, a random 96-bit nonce, and the tenant id as associated
data, so a sealed secret copied into another tenant's row does not open. Changing
`KALOS_SECRET_KEY` makes stored secrets unreadable: each tenant reconnects.
"""
from __future__ import annotations

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

SECRET_KEY_VAR = "KALOS_SECRET_KEY"  # noqa: S105 - the variable NAME, not a secret
MIN_KEY_CHARS = 32
_PREFIX = "v1."
_SALT = b"kalos-integrations-v1"


class SecretKeyUnavailable(RuntimeError):
    """`KALOS_SECRET_KEY` is unset or shorter than MIN_KEY_CHARS."""


class SecretUnreadable(RuntimeError):
    """A sealed value could not be opened (another key, another tenant, or damaged)."""


def key_configured() -> bool:
    return len(os.environ.get(SECRET_KEY_VAR, "").strip()) >= MIN_KEY_CHARS


def _key(purpose: str) -> bytes:
    raw = os.environ.get(SECRET_KEY_VAR, "").strip()
    if len(raw) < MIN_KEY_CHARS:
        raise SecretKeyUnavailable(f"{SECRET_KEY_VAR} must be set to at least {MIN_KEY_CHARS} characters")
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=_SALT, info=purpose.encode()).derive(raw.encode())


def _aad(purpose: str, tenant: str) -> bytes:
    return f"{purpose}\x00{tenant}".encode()


def seal(plaintext: str, *, tenant: str, purpose: str = "benchling-client-secret") -> str:
    nonce = os.urandom(12)
    ct = AESGCM(_key(purpose)).encrypt(nonce, plaintext.encode(), _aad(purpose, tenant))
    return _PREFIX + base64.urlsafe_b64encode(nonce + ct).decode()


def open_sealed(sealed: str, *, tenant: str, purpose: str = "benchling-client-secret") -> str:
    if not isinstance(sealed, str) or not sealed.startswith(_PREFIX):
        raise SecretUnreadable("stored secret has an unknown format")
    try:
        blob = base64.urlsafe_b64decode(sealed[len(_PREFIX):].encode())
        return AESGCM(_key(purpose)).decrypt(blob[:12], blob[12:], _aad(purpose, tenant)).decode()
    except (InvalidTag, ValueError):
        raise SecretUnreadable("stored secret cannot be opened with the current key") from None
