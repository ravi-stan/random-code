# tests/unit/fixtures/redis_unit.py
from __future__ import annotations
import uuid
from functools import partial
import pytest
from fakeredis.aioredis import FakeRedis, FakeServer
from redis.asyncio import Redis

# ---------- Core: shared fake server ----------

@pytest.fixture(scope="session")
def _fake_redis_server() -> FakeServer:
    """One in-memory Redis server for the entire session (shared by clients)."""
    return FakeServer()

# ---------- Clients ----------

@pytest.fixture
async def redis_client(_fake_redis_server) -> Redis:
    """Binary-safe client (bytes in/out), flushed before and after each test."""
    client: Redis = FakeRedis(server=_fake_redis_server, decode_responses=False)
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.close()

@pytest.fixture
async def redis_text_client(_fake_redis_server) -> Redis:
    """Text client (str in/out), useful for tests that expect decoded responses."""
    client: Redis = FakeRedis(server=_fake_redis_server, decode_responses=True)
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.close()

# ---------- Namespacing ----------

@pytest.fixture
def redis_namespace(worker_id) -> str:
    """Unique namespace per test, with a worker suffix for xdist friendliness."""
    suffix = "" if worker_id in ("master", "gw0") else f":{worker_id}"
    return f"test:{uuid.uuid4().hex}{suffix}"

def _ns(ns: str, *parts: str) -> str:
    return ns + ":" + ":".join(parts)

@pytest.fixture
def make_key(redis_namespace):
    """Curried key builder: make_key('user','42') -> 'ns:user:42'."""
    return partial(_ns, redis_namespace)

# ---------- App-level cache (optional) ----------

@pytest.fixture
async def cache(redis_client, redis_namespace):
    """
    If your app exposes a cache wrapper, return an instance bound to this test's namespace.
    """
    try:
        from app.cache import Cache, CacheConfig
    except Exception:
        # Not all repos will have this; make it optional.
        pytest.skip("app.cache not available")
    cache = Cache(redis_client, CacheConfig(namespace=redis_namespace, default_ttl=2, compress_threshold=512))
    try:
        yield cache
    finally:
        await cache.clear_namespace()

# ---------- Ensure app code under test uses the same fake server ----------

@pytest.fixture
def patch_redis_from_url(monkeypatch, _fake_redis_server):
    """Force Redis.from_url(...) to return a FakeRedis bound to the same FakeServer."""
    import redis.asyncio as redis_asyncio
    import redis.asyncio.client as redis_client_mod

    def _from_url(cls, url, **kwargs):
        return FakeRedis(server=_fake_redis_server, decode_responses=kwargs.get("decode_responses", False))

    monkeypatch.setattr(redis_asyncio.Redis, "from_url", classmethod(_from_url))
    monkeypatch.setattr(redis_client_mod.Redis, "from_url", classmethod(_from_url))

