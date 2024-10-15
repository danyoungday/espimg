import flappy_bird_gymnasium
import gymnasium
import numpy as np
import matplotlib.pyplot as plt

def collect_data(n_rollouts: int, n_envs: int, n_steps: int):
    """
    Rollouts: (N, L, 4)
    """
    env = gymnasium.make_vec("FlappyBird-v0", num_envs=n_envs, use_lidar=False)
    all_rollouts = []
    n_roll = n_rollouts // n_envs
    for _ in range(n_roll):
        rollouts = [[] for _ in range(n_envs)]
        dones = np.array([False] * n_envs)

        seeds = [int(seed) for seed in np.random.randint(0, 2**32-1, n_envs)]
        obses, _ = env.reset(seed=seeds)
        i = 0
        while not np.all(dones) and i < n_steps:
            # Next action:
            # (feed the observation to your agent here)
            actions = np.random.choice([0, 1], n_envs, p=[0.9, 0.1])

            # Processing:
            next_obses, rewards, terminateds, _, infos = env.step(actions)

            # This is done before we update dones so that we can get the last step
            for i, state in enumerate(zip(obses, actions, rewards, terminateds, next_obses, dones)):
                obs, action, reward, terminated, next_obs, done = state
                if not done:
                    rollouts[i].append((obs, [action], [reward], [int(terminated)], next_obs))

            dones = np.logical_or(dones, terminateds)
            obses = next_obses
            i += 1

        all_rollouts.extend(rollouts)

    env.close()

    return all_rollouts


# def collect_data(n_rollouts: int, n_envs: int, n_steps: int):
#     """
#     Rollouts: (N, L, 4)
#     """
#     env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
#     all_rollouts = []
#     n_roll = n_rollouts // n_envs
#     for _ in range(n_roll):
#         rollout = []
#         done = False

#         seed = np.random.randint(0, 2**32-1)
#         obs, _ = env.reset(seed=seed)
#         i = 0
#         while not done and i < n_steps:
#             # Next action:
#             # (feed the observation to your agent here)
#             action = np.random.choice([0, 1], 1, p=[0.9, 0.1])

#             # Processing:
#             next_obs, reward, terminated, _, info = env.step(action)

#             # This is done before we update dones so that we can get the last step
#             if not done:
#                 rollout.append((obs, [action], [reward], [int(terminated)], next_obs))

#             done = done or terminated
#             obs = next_obs
#             i += 1

#         all_rollouts.append(rollout)

#     env.close()

#     return all_rollouts