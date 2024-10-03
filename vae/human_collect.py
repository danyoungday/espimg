"""
Collect human data for training VAE using built in play function.
Keys are bound to WASD.
Press esc to stop collecting data.
"""
import gymnasium
from gymnasium.utils.play import play
import numpy as np


def record_states(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    """
    Function called at every step of the process.
    * obs_t: observation before performing action
    * obs_tp1: observation after performing action
    * action: action that was executed
    * rew: reward that was received
    * terminated: whether the environment is terminated or not
    * truncated: whether the environment is truncated or not
    * info: debug info
    """
    states.append(obs_t)


def process_data(states: list[np.ndarray]) -> np.ndarray:
    """
    Processes states to only include valid states.
    """
    p_states = []
    for state in states:
        if isinstance(state, np.ndarray) and state.shape == (96, 96, 3):
            p_states.append(state)

    return np.array(p_states)


def main():
    """
    Runs the play function to collect human data.
    """
    global states
    states = []
    env = gymnasium.make("CarRacing-v2", render_mode="rgb_array", continuous=False)
    play(env, keys_to_action={"0": 0, "d": 1, "a": 2, "w": 3, "s": 4}, callback=record_states)

    processed = process_data(states)
    print("Collected ", len(processed), " states")
    np.save("data/CarRacing-v2/human-test.npy", processed)


if __name__ == "__main__":
    main()