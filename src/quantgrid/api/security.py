"""API-key authentication and per-user rate limiting.

Keys are configured as ``QG_API_KEYS="<key>:<user_id>,..."`` and compared in constant
time. Rate limits are fixed one-minute windows per (bucket, user), counted in Redis so
they hold across API replicas; an in-memory limiter is used in tests or when Redis is
unavailable (fail-open with a logged warning: rate limiting protects capacity, not
integrity - quotas are enforced in PostgreSQL regardless).
"""

from __future__ import annotations

import hmac
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from quantgrid.domain.errors import RateLimited, Unauthorized

log = logging.getLogger("quantgrid.api.security")


@dataclass(frozen=True)
class Principal:
    user_id: str


class ApiKeyAuthenticator:
    def __init__(self, key_map: dict[str, str]) -> None:
        self._keys = list(key_map.items())

    def authenticate(self, presented: str | None) -> Principal:
        if not presented:
            raise Unauthorized("missing X-API-Key header")
        match: str | None = None
        for key, user in self._keys:  # constant-time compare against every key
            if hmac.compare_digest(key.encode(), presented.encode()):
                match = user
        if match is None:
            raise Unauthorized("invalid API key")
        return Principal(user_id=match)


class RateLimiter(Protocol):
    def hit(self, bucket: str, user_id: str, limit: int) -> None:
        """Count one request; raise :class:`RateLimited` when over ``limit`` per minute."""


class InMemoryRateLimiter:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str, int], int] = {}

    def hit(self, bucket: str, user_id: str, limit: int) -> None:
        now = self._clock()
        window = int(now // 60)
        with self._lock:
            key = (bucket, user_id, window)
            self._counts[key] = self._counts.get(key, 0) + 1
            count = self._counts[key]
            for stale in [k for k in self._counts if k[2] < window - 1]:
                del self._counts[stale]
        if count > limit:
            raise RateLimited(
                f"rate limit of {limit} requests/minute exceeded for {bucket}",
                retry_after_s=int(60 - (now % 60)) + 1,
            )


class RedisRateLimiter:
    def __init__(self, redis_url: str, *, fallback: RateLimiter | None = None) -> None:
        import redis

        self._redis = redis.Redis.from_url(
            redis_url, socket_timeout=0.5, socket_connect_timeout=0.5
        )
        self._fallback = fallback or InMemoryRateLimiter()

    def hit(self, bucket: str, user_id: str, limit: int) -> None:
        import redis

        now = time.time()
        window = int(now // 60)
        key = f"qg:rl:{bucket}:{user_id}:{window}"
        try:
            pipe = self._redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, 120)
            count = int(pipe.execute()[0])
        except redis.RedisError:
            log.warning("rate limiter: redis unavailable, using in-process fallback")
            self._fallback.hit(bucket, user_id, limit)
            return
        if count > limit:
            raise RateLimited(
                f"rate limit of {limit} requests/minute exceeded for {bucket}",
                retry_after_s=int(60 - (now % 60)) + 1,
            )
