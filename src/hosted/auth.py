"""Scrypt passwords and random bearer sessions; no browser ambient credentials."""
from dataclasses import dataclass
import hashlib
import hmac
import secrets
from fastapi import HTTPException, Request


@dataclass(frozen=True)
class Principal:
    account_id: str
    email: str


def hash_password(password: str) -> str:
    if not 12 <= len(password) <= 1024:
        raise ValueError('Password must contain between 12 and 1024 characters')
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=32768, r=8, p=3, dklen=32, maxmem=64 * 1024 * 1024)
    return f'scrypt-v1${salt.hex()}${digest.hex()}'


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, salt, digest = encoded.split('$')
        if algorithm != 'scrypt-v1' or len(password) > 1024:
            return False
        candidate = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=32768, r=8, p=3, dklen=32, maxmem=64 * 1024 * 1024)
        return hmac.compare_digest(candidate, bytes.fromhex(digest))
    except (ValueError, TypeError):
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def bearer_token(request: Request) -> str:
    scheme, _, token = request.headers.get('Authorization', '').partition(' ')
    if scheme.lower() != 'bearer' or not 32 <= len(token) <= 256:
        raise HTTPException(401, 'Authentication required', headers={'WWW-Authenticate': 'Bearer'})
    return token


def require_principal(request: Request) -> Principal:
    account = request.app.state.store.authenticate(bearer_token(request))
    if account is None:
        raise HTTPException(401, 'Invalid or expired session', headers={'WWW-Authenticate': 'Bearer'})
    return Principal(account['id'], account['email'])
