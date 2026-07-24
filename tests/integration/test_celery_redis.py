from celery import Celery
from celery.contrib.testing.worker import start_worker
from testcontainers.redis import RedisContainer  # type: ignore


def test_celery_redis_enqueue_consume_result(redis_container: RedisContainer) -> None:
    """Live round-trip over the Celery Redis broker + result backend, exercising the
    Kombu redis transport with the pinned redis-py (must stay within Kombu's supported
    range). Covers enqueue (broker publish), consume (worker), and result retrieval
    (redis backend)."""
    url = (
        f"redis://{redis_container.get_container_host_ip()}:"
        f"{redis_container.get_exposed_port(6379)}"
    )
    app = Celery(
        "ragpi_celery_redis_test",
        broker=url,
        backend=url,
        broker_connection_retry_on_startup=True,
    )

    @app.task(name="ragpi_test.add")
    def add(a: int, b: int) -> int:
        return a + b

    with start_worker(app, perform_ping_check=False, loglevel="error"):
        async_result = add.delay(2, 3)
        assert async_result.get(timeout=30) == 5
        # Result was stored in and read back from the Redis backend.
        assert async_result.successful()
