"""Seeded, simulated datagrams with explicit delivery service time (no I/O)."""
from copy import deepcopy
from dataclasses import dataclass
import heapq
import math
import random


@dataclass(frozen=True)
class LinkConfig:
    uplink_delay_s: float = .004
    downlink_delay_s: float = .004
    jitter_s: float = 0.
    loss_probability: float = 0.
    seed: int = 20261023

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in
               (self.uplink_delay_s, self.downlink_delay_s, self.jitter_s, self.loss_probability)):
            raise ValueError('Link bounds must be finite and nonnegative')
        if self.loss_probability > 1 or type(self.seed) is not int or self.seed < 0:
            raise ValueError('Invalid loss probability or seed')


@dataclass(frozen=True)
class Delivery:
    direction: str
    payload: object
    sent_s: float
    due_s: float
    delivered_s: float


class SimulatedTransport:
    def __init__(self, config=None):
        self.config = config or LinkConfig()
        self.rng = random.Random(self.config.seed)
        self._queue = []
        self._counter = 0
        self.events = []

    def send(self, direction, payload, *, now_s, delay_s=None, drop=False):
        if direction not in ('uplink', 'downlink') or not math.isfinite(now_s) or now_s < 0:
            raise ValueError('Invalid link direction or time')
        c = self.config
        delay = (getattr(c, direction+'_delay_s') + self.rng.uniform(-c.jitter_s, c.jitter_s)
                 if delay_s is None else delay_s)
        if not math.isfinite(delay) or (delay_s is not None and delay < 0):
            raise ValueError('Invalid explicit delay')
        lost = drop or self.rng.random() < c.loss_probability
        event = dict(kind='dropped' if lost else 'scheduled', direction=direction,
                     packet_id=self._counter, sent_s=now_s, due_s=now_s+max(0., delay))
        self.events.append(event)
        if not lost:
            heapq.heappush(self._queue, (event['due_s'], self._counter,
                direction, deepcopy(payload), now_s))
        self._counter += 1

    def drain(self, now_s):
        if not math.isfinite(now_s) or now_s < 0:
            raise ValueError('Invalid delivery time')
        result = []
        while self._queue and self._queue[0][0] <= now_s+1e-10:
            due, ident, direction, payload, sent = heapq.heappop(self._queue)
            result.append(Delivery(direction, payload, sent, due, now_s))
            self.events.append(dict(kind='delivered', direction=direction, packet_id=ident,
                sent_s=sent, due_s=due, delivered_s=now_s))
        return result
