from typing import Callable, Optional

from tianshou.data import Collector
from tianshou.policy import BasePolicy


def evaluate_policy(
    policy: BasePolicy,
    collector: Collector,
    n_episode: int,
    test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
    timestamp: Optional[dict] = None,
    collect_kwargs={},
):
    collector.reset_buffer()
    policy.eval()
    if test_fn is not None:
        test_fn(timestamp)

    result = collector.collect(n_episode=n_episode, **collect_kwargs)
    return result
