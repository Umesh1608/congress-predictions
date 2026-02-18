import redis.asyncio as aioredis

from src.config import settings

redis_pool = aioredis.ConnectionPool.from_url(settings.redis_url, decode_responses=True)


def get_redis() -> aioredis.Redis:
    return aioredis.Redis(connection_pool=redis_pool)
