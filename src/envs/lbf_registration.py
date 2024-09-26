import warnings

from gym.envs.registration import register

warnings.filterwarnings("ignore", category=UserWarning)

register(
    id="Foraging-8x8-5p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 5,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 1,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-11x11-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 2,
        "sight": 11,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-15x15-3p-4f-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 4,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": False,
    },
)

register(
    id="Foraging-15x15-3p-4f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 4,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8s-25x25-8p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 8,
        "max_player_level": 3,
        "field_size": (25, 25),
        "max_food": 5,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5s-25x25-8p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 8,
        "max_player_level": 3,
        "field_size": (25, 25),
        "max_food": 5,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-50x50-8p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 8,
        "max_player_level": 3,
        "field_size": (50, 50),
        "max_food": 5,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-30x30-7p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 7,
        "max_player_level": 3,
        "field_size": (30, 30),
        "max_food": 5,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-30x30-7p-4f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 7,
        "max_player_level": 3,
        "field_size": (30, 30),
        "max_food": 4,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-4s-30x30-8p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 8,
        "max_player_level": 3,
        "field_size": (30, 30),
        "max_food": 5,
        "sight": 4,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-15x15-5p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 5,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 3,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-11x11-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-4s-11x11-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 2,
        "sight": 4,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-9x9-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (9, 9),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-9x9-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (9, 9),
        "max_food": 2,
        "sight": 9,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-8x8-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-6x6-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (6, 6),
        "max_food": 2,
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-15x15-3p-5f-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 5,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": False,
    },
)

register(
    id="Foraging-6x6-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (6, 6),
        "max_food": 1,
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7x7-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (7, 7),
        "max_food": 1,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7x7-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (7, 7),
        "max_food": 2,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-7x7-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (7, 7),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-4p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-4p-2f-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": False,
    },
)

register(
    id="Foraging-8x8-4p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 1,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5x5-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (5, 5),
        "max_food": 2,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5x5-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (5, 5),
        "max_food": 1,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-6p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 6,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 1,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-15x15-3p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 5,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-10x10-4p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (10, 10),
        "max_food": 1,
        "sight": 10,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7x7-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (7, 7),
        "max_food": 3,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-9x9-4p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (9, 9),
        "max_food": 2,
        "sight": 9,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-20x20-5p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 5,
        "max_player_level": 3,
        "field_size": (20, 20),
        "max_food": 3,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5s-20x20-5p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 5,
        "max_player_level": 3,
        "field_size": (20, 20),
        "max_food": 3,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-11x11-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 3,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-4s-11x11-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 3,
        "sight": 4,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-13x13-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 3,
        "sight": 13,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-11x11-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 3,
        "sight": 11,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-9x9-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (9, 9),
        "max_food": 3,
        "sight": 9,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5x5-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (5, 5),
        "max_food": 3,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-10x10-3p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (10, 10),
        "max_food": 3,
        "sight": 10,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-12x12-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (12, 12),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-6s-12x12-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (12, 12),
        "max_food": 2,
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-12x12-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (12, 12),
        "max_food": 2,
        "sight": 12,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 1,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)