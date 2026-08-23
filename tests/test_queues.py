from src.voice_dsp.queues import BoundedDropOldestQueue


def test_queue_never_exceeds_capacity_when_fed_faster_than_drained():
    q = BoundedDropOldestQueue(capacity=8)

    for i in range(1000):
        q.put(i)
        assert len(q) <= 8
        if i % 3 == 0:  # drain slower than we feed
            q.get()

    assert len(q) <= 8


def test_queue_drops_oldest_not_newest():
    q = BoundedDropOldestQueue(capacity=4)
    for i in range(10):
        q.put(i)

    # capacity 4, fed 0..9 with no draining -> should hold the newest
    # four items (6,7,8,9), oldest ones (0..5) dropped.
    remaining = []
    while True:
        item = q.get()
        if item is None:
            break
        remaining.append(item)

    assert remaining == [6, 7, 8, 9]
    assert q.dropped_count == 6


def test_queue_capacity_one():
    q = BoundedDropOldestQueue(capacity=1)
    q.put("a")
    q.put("b")
    q.put("c")
    assert len(q) == 1
    assert q.get() == "c"
    assert q.get() is None


def test_queue_empty_get_returns_none():
    q = BoundedDropOldestQueue(capacity=4)
    assert q.get() is None
    assert len(q) == 0
