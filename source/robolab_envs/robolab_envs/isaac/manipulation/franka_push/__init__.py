import gymnasium as gym

from .variants.image_push_env_cfg import FrankaPushImageEnvCfg  # noqa: F401

##
# Register environments in the gym namespace
##

gym.register(
    id="Robolab-Isaac-FrankaPush-Image-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaPushImageEnvCfg,
    },
)
