from gym.envs.registration import register


register(
    id="AntTruncatedObs-v2",
    entry_point="pytorch_mbpo.envs:AntTruncatedObsEnv",
    max_episode_steps=1000
)

register(
    id="HumanoidTruncatedObs-v2",
    entry_point="pytorch_mbpo.envs:HumanoidTruncatedObsEnv",
    max_episode_steps=1000
)
