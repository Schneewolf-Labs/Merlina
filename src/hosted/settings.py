"""Explicit hosted settings; credentials belong to operators, never job payloads."""
from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class HostedSettings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv('MERLINA_HOSTED_DATA_DIR', './data/hosted')))
    database_path: Path | None = None
    session_ttl_seconds: int = 86400
    cors_origins: tuple[str, ...] = ()
    paid_execution_enabled: bool = False
    stripe_secret_key: str | None = None
    stripe_webhook_secret: str | None = None
    checkout_success_url: str | None = None
    checkout_cancel_url: str | None = None
    billing_currency: str = 'usd'
    billing_policy_version: str | None = None
    billing_failure_policy: str | None = None
    billing_overage_policy: str | None = None
    max_dataset_bytes: int = 10 * 1024 * 1024

    @classmethod
    def from_env(cls):
        prefix = 'MERLINA_HOSTED_'
        values = {}
        for key in ('data_dir', 'database_path', 'stripe_secret_key',
                    'stripe_webhook_secret', 'checkout_success_url', 'checkout_cancel_url',
                    'billing_currency', 'billing_policy_version', 'billing_failure_policy',
                    'billing_overage_policy'):
            value = os.getenv(prefix + key.upper())
            if value is not None:
                values[key] = value
        for key in ('session_ttl_seconds', 'max_dataset_bytes'):
            value = os.getenv(prefix + key.upper())
            if value is not None:
                values[key] = int(value)
        enabled = os.getenv(prefix + 'PAID_EXECUTION_ENABLED')
        if enabled is not None:
            if enabled.lower() not in ('true', 'false'):
                raise ValueError('MERLINA_HOSTED_PAID_EXECUTION_ENABLED must be true or false')
            values['paid_execution_enabled'] = enabled.lower() == 'true'
        origins = os.getenv(prefix + 'CORS_ORIGINS')
        if origins is not None:
            values['cors_origins'] = tuple(item.strip() for item in origins.split(',') if item.strip())
        return cls(**values)

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.database_path = Path(self.database_path) if self.database_path else self.data_dir / 'hosted.sqlite3'
        if not 1 <= self.max_dataset_bytes <= 100 * 1024 * 1024:
            raise ValueError('Dataset limit must be positive and at most 100 MiB')
        if self.billing_currency != 'usd':
            raise ValueError('Only USD billing is supported in the pilot')
        if self.session_ttl_seconds < 60 or self.session_ttl_seconds > 604800:
            raise ValueError('Session TTL must be between 60 seconds and seven days')
        if any(origin == '*' or not origin.startswith(('https://', 'http://localhost:')) for origin in self.cors_origins):
            raise ValueError('Hosted CORS requires explicit HTTPS origins (localhost allowed for development)')
