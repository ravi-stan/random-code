from __future__ import annotations
import uuid
import pytest

# fakeredis (sync) — use the sync, not aioredis, classes
from fakeredis import FakeRedis, FakeServer

# redis-py (sync) classes you use in your app
import redis
import redis.cluster as redis_cluster


# -------- Shared in-memory server for the whole session --------

@pytest.fixture(scope="session")
def _fake_redis_server() -> FakeServer:
    """
    One in-memory Redis server for the entire test session.
    All FakeRedis clients will talk to this server.
    """
    return FakeServer()


# -------- Plain Redis client (sync) --------

@pytest.fixture
def redis_client(_fake_redis_server) -> redis.Redis:
    """
    Fresh sync Redis client per test (binary-safe).
    DB is flushed before and after each test to prevent cross-test pollution.
    """
    client = FakeRedis(server=_fake_redis_server, decode_responses=False)
    client.flushdb()
    try:
        yield client
    finally:
        client.flushdb()
        client.close()


# -------- Optional: simple namespacing helpers --------

@pytest.fixture
def redis_namespace() -> str:
    """A unique prefix per test so parallel tests cannot collide."""
    return f"test:{uuid.uuid4().hex}"

def _ns(ns: str, *parts: str) -> str:
    return ns + ":" + ":".join(parts)

@pytest.fixture
def make_key(redis_namespace):
    """Curried key builder: make_key('user','42') -> 'ns:user:42'."""
    from functools import partial
    return partial(_ns, redis_namespace)


# -------- Patch redis.Redis.from_url and RedisCluster for unit tests --------

@pytest.fixture
def patch_redis_from_url(monkeypatch, _fake_redis_server):
    """
    Force redis.Redis.from_url(...) and redis.cluster.RedisCluster.from_url(...) to return
    fakes bound to the same FakeServer.

    NOTE: For RedisCluster we return a small proxy that delegates to a FakeRedis instance.
    This is sufficient for most unit tests that don't require cluster-specific behavior.
    """
    # 1) Plain Redis.from_url -> FakeRedis
    def _from_url(cls, url, **kwargs):
        return FakeRedis(
            server=_fake_redis_server,
            decode_responses=kwargs.get("decode_responses", False),
        )
    monkeypatch.setattr(redis.Redis, "from_url", classmethod(_from_url))

    # 2) RedisCluster: provide a proxy that looks like a client but delegates to FakeRedis
    class _FakeRedisClusterProxy:
        """
        Minimal stand-in for redis.cluster.RedisCluster that forwards all attributes
        to a single-node FakeRedis. Good enough for unit tests that don't rely on MOVED/ASK,
        slot hashing, cross-slot pipelines, etc.
        """
        def __init__(self, **kwargs):
            self._client = FakeRedis(
                server=_fake_redis_server,
                decode_responses=kwargs.get("decode_responses", False),
            )

        @classmethod
        def from_url(cls, url, **kwargs):
            return cls(**kwargs)

        # Delegate everything else to the underlying FakeRedis
        def __getattr__(self, name):
            return getattr(self._client, name)

        # Optional: context manager support if your code uses `with client.pipeline():`
        def pipeline(self, *args, **kwargs):
            return self._client.pipeline(*args, **kwargs)

        def close(self):
            return self._client.close()

    # Two patches to maximize compatibility:
    #   a) Replace the class symbol (affects imports done AFTER this patch)
    #   b) Patch from_url (affects code that uses from_url factory)
    monkeypatch.setattr(redis_cluster, "RedisCluster", _FakeRedisClusterProxy)
    if hasattr(redis_cluster.RedisCluster, "from_url"):
        monkeypatch.setattr(redis_cluster.RedisCluster, "from_url", classmethod(_FakeRedisClusterProxy.from_url))

    # yield so tests can run with patch active
    yield




@pytest.mark.usefixtures("patch_redis_from_url")
def test_uses_from_url_and_basic_ops(redis_client, make_key):
    # Suppose your code under test does: redis.Redis.from_url("redis://...").get(...)
    import redis
    c = redis.Redis.from_url("redis://unit")
    k = make_key("user", "42")
    c.set(k, b"hello", ex=60)
    assert c.get(k) == b"hello"

@pytest.mark.usefixtures("patch_redis_from_url")
def test_redis_cluster_proxy_behaves_like_simple_client(make_key):
    # Suppose your code under test does: RedisCluster.from_url("redis://cluster")
    from redis.cluster import RedisCluster
    rc = RedisCluster.from_url("redis://cluster")
    k = make_key("cart", "A1")
    rc.hset(k, mapping={"sku": "123", "qty": "2"})
    assert rc.hgetall(k) == {b"sku": b"123", b"qty": b"2"}

