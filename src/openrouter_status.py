from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from dotenv import dotenv_values

DEFAULT_ENV_FILE = Path(".env")
OPENROUTER_KEY_NAME = "OPENROUTER_API_KEY"
DISPLAY_TIMEZONE_NAME = "OPENROUTER_STATUS_TIMEZONE"
OPENROUTER_STATUS_URL = "https://openrouter.ai/api/v1/key"


class StatusError(RuntimeError):
    """Raised when the key status cannot be read or displayed."""


def _format_expiry(expires_at: str, display_timezone: str | None) -> str:
    """Format an expiry in the configured timezone.

    Args:
        expires_at: ISO 8601 API-key expiry.
        display_timezone: IANA timezone used to display the expiry.

    Returns:
        A human-readable local expiry.

    Raises:
        StatusError: If the expiry or timezone is invalid.
    """
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        if expiry.utcoffset() is None:
            raise ValueError
        zone = ZoneInfo(display_timezone) if display_timezone else UTC
    except (ValueError, ZoneInfoNotFoundError) as error:
        raise StatusError("Invalid expiry configuration") from error
    local_expiry = expiry.astimezone(zone)
    return f"{local_expiry:%A}, {local_expiry.day} {local_expiry:%B %Y at %H:%M %Z}"


def status() -> None:
    """Show spend and expiry for the API key stored in ``.env``."""
    configuration = dotenv_values(DEFAULT_ENV_FILE)
    api_key = configuration.get(OPENROUTER_KEY_NAME)
    display_timezone = configuration.get(DISPLAY_TIMEZONE_NAME)
    if not isinstance(api_key, str) or not api_key:
        sys.stderr.write("Status failed: Set OPENROUTER_API_KEY in .env first (see .env.example)\n")
        raise SystemExit(1)

    try:
        response = requests.get(
            OPENROUTER_STATUS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            allow_redirects=False,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()["data"]
        usage = float(data["usage"])
        limit = float(data["limit"])
        remaining = float(data["limit_remaining"])
        expires_at = data.get("expires_at")
        if expires_at is not None and not isinstance(expires_at, str):
            raise TypeError
        expiry = (
            datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expires_at
            else None
        )
        if expiry is not None and expiry.utcoffset() is None:
            raise ValueError
        if display_timezone is not None and not isinstance(display_timezone, str):
            raise TypeError
        formatted_expiry = (
            _format_expiry(expires_at, display_timezone) if expires_at else "Not set"
        )
    except (
        StatusError,
        requests.RequestException,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        unauthorized = (
            isinstance(error, requests.HTTPError)
            and error.response is not None
            and error.response.status_code == 401
        )
        message = (
            "The API key is invalid, disabled, or expired"
            if unauthorized
            else "OpenRouter could not return the key status"
        )
        sys.stderr.write(f"Status failed: {message}\n")
        raise SystemExit(1) from error

    sys.stdout.write("OpenRouter API key status\n")
    if expiry:
        seconds_left = (expiry - datetime.now(UTC)).total_seconds()
        if seconds_left <= 0:
            time_left = "Expired"
        else:
            total_hours = int(seconds_left // 3_600)
            days, hours = divmod(total_hours, 24)
            time_left = (
                f"{days} days, {hours} hours"
                if days
                else f"{hours} hours" if hours else "Less than 1 hour"
            )
    else:
        time_left = "Not set"
    sys.stdout.write(f"API access remaining: {time_left}\n")
    sys.stdout.write(f"Spend remaining: ${remaining:.2f} of ${limit:.2f}\n")
    sys.stdout.write(f"Spend used: ${usage:.2f}\n")
    sys.stdout.write(f"API access expires: {formatted_expiry}\n")

