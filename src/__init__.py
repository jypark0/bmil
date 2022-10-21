from gym import register

# Fetch Robotics
for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    kwargs = {
        "reward_type": reward_type,
    }

    register(
        id="PickAndPlace{}-v2".format(suffix),
        entry_point="src.envs.fetch_env:MyFetchPickAndPlaceEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id="Push{}-v2".format(suffix),
        entry_point="src.envs.fetch_env:MyFetchPushEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )

# Adroit Relocate
register(
    id="AdroitRelocate-v0",
    entry_point="src.envs.adroit_env:MyRelocateEnvV0",
    max_episode_steps=200,
)
