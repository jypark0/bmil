import importlib
from typing import Optional, Sequence

import gym


def make_env(
    env_id: str,
    wrappers: Sequence = [],
    seed: Optional[int] = None,
    env_kwargs: dict = {},
):
    def _init():
        env = gym.make(env_id, **env_kwargs)
        for wrapper in wrappers:
            # Get first key (wrapper name)
            wrapper_name = list(wrapper)[0]
            # Get module
            module_name, class_name = wrapper_name.rsplit(".", 1)
            wrapper_cls = getattr(importlib.import_module(module_name), class_name)

            # Get kwargs
            kwargs = wrapper[wrapper_name]
            env = wrapper_cls(env, **kwargs)

        # RecordEpisodeStatistics
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Seed
        if seed:
            env.seed(seed)
            env.action_space.seed(seed)
        else:
            env.seed()
            env.action_space.seed()

        return env

    return _init
