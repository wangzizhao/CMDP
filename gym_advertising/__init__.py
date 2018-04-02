from gym.envs.registration import register

register(
    id="advertising-v0",
    entry_point="gym_advertising.envs:AdvertisingEnv",
)
