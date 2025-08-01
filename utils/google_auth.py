# Third-Party Library
import functools
from pathlib import Path
from typing import Set

from fastapi import HTTPException, Request, status
from fastapi.security.utils import get_authorization_scheme_param
from google.auth.transport import requests as grequests
from google.oauth2 import id_token

ALLOWED_PEOPLE_FILE: Path = Path(__file__).parent / "allowed_people.txt"


@functools.lru_cache(maxsize=1)
def get_allowed_emails() -> Set[str]:
    """
    Load and cache the set of allowed emails from file.
    """
    try:
        with ALLOWED_PEOPLE_FILE.open("r") as f:
            return {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Allowed people file not found",
        )


async def verify_company_email(request: Request) -> None:
    """
    Dependency to verify that the incoming request has a valid Google OAuth2 token
    and that the email is authorized.
    """
    auth = request.headers.get("Authorization")
    if not auth:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    scheme, token = get_authorization_scheme_param(auth)
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth scheme")
    try:
        idinfo = id_token.verify_oauth2_token(token, grequests.Request())
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Google ID token"
        ) from exc

    email = idinfo.get("email")
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No email in token")

    allowed_emails = get_allowed_emails()
    if email.lower() not in allowed_emails:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Email not allowed")
