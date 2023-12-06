from pathlib import Path
from typing import NamedTuple
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from tqdm import tqdm  # smart progress meter
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme()


class Params(NamedTuple):  # https://docs.python.org/3/library/collections.html#collections.namedtuple
    episodes: int  # Episode = "1 game run (e.g. until win or lose or maximum "play time" reached)
    lr: float  # Learning rate ("how fast" the agent adapts to changes---learns)
    gamma: float  # Discounting rate ("how much value" to give to future vs immediate rewards)
    e: float  # Exploration probability (e.g. do random action instead of arg_max(Q_values)
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    runs: int  # 1 experiment (run) is a series of episodes, we need more runs to get a good estimate of the performance (e.g. averaging random effects)
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    prob_frozen: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved


params = Params(
    episodes=250_000,
    lr=0.1,  # learn "slowly" (=do not change Q-values too much at once)
    gamma=0.9,  # give value to future rewards ('cause we need many steps to reach the destination!)
    e=0.1,
    map_size=5,
    seed=123,
    is_slippery=False,
    runs=3,
    action_size=None,
    state_size=None,
    prob_frozen=0.9,
    savefig_folder=Path("res/frozenlake/"),
)

rng = np.random.default_rng(params.seed)
params.savefig_folder.mkdir(parents=True, exist_ok=True)  # Create the figure folder if it doesn't exist


class QLearningAgent:
    def __init__(self,
                 lr,
                 gamma,
                 state_s,
                 action_s):
        self.lr = lr
        self.gamma = gamma
        self.state_s = state_s
        self.action_s = action_s
        self.qtable = self.init_qtable()

    def update(self, state, action, reward, next_state):
        """Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
                reward
                + self.gamma * np.max(self.qtable[next_state, :])
                - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.lr * delta
        return q_update

    def init_qtable(self):
        return np.zeros((self.state_s, self.action_s))


class EpsilonGreedy:
    """Exploration strategy"""
    def __init__(self, e):  # exploration probability
        self.e = e

    def choose_action(self, action_s, state, qtable):
        rnd = rng.uniform(0, 1)
        if rnd < self.e:
            action = action_s.sample()  # random action (e.g. e=0.1 means 10% of the time)
        else:
            if np.all(qtable[state, :] == qtable[state, 0]):  # If all actions are the same for this state we choose a random one (otherwise `np.argmax()` would always take the first one)
                action = action_s.sample()
            else:
                action = np.argmax(qtable[state, :])  # "greedy" action (= the "best" as far as we currently know)
        return action


def run_env():
    """Run 1 experiment (= 1 whole series of episodes)"""
    # data tracking for measuring performance
    rewards = np.zeros((params.episodes, params.runs))
    steps = np.zeros((params.episodes, params.runs))
    episodes = np.arange(params.episodes)
    qtables = np.zeros((params.runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.runs):  # 1 run is one experiment (series of episodes)
        learner.init_qtable()

        for episode in tqdm(
            episodes, desc=f"Run {run + 1}/{params.runs} - Episodes", leave=False
        ):
            state, _ = env.reset(seed=params.seed)
            step = 0
            done = False
            total_rewards = 0

            while not done:  # the Gymnasium environment itself tells us when it's done (episode ended)
                action = explorer.choose_action(
                    action_s=env.action_space, state=state, qtable=learner.qtable
                )

                all_states.append(state)
                all_actions.append(action)

                next_s, rew, term, trunc, _ = env.step(action)  # see Gymnasium API
                done = term or trunc  # either the episode ended or we reached the maximum number of steps

                learner.qtable[state, action] = learner.update(
                    state, action, rew, next_s
                )

                total_rewards += rew
                step += 1  # keep track of steps needed to reach the goal
                state = next_s
            rewards[episode, run] = total_rewards
            steps[episode, run] = step
        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions

#############################################
# CODE BELOW IS JUST FOR NICE VISUALIZATION #
#############################################


def to_pandas(episodes, params, rewards, steps, map_size):
    """Convert the results of the simulation in Pandas dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.runs),
            "Rewards": rewards.flatten(),
            "Steps": steps.flatten()
        }
    )
    res["cumul_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st


def plot_steps_and_rewards(rewards_df, steps_df):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(
        data=rewards_df, x="Episodes", y="cumul_rewards", hue="map_size", ax=ax[0]
    )
    ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=steps_df, x="Episodes", y="Steps", hue="map_size", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")

    for axi in ax:
        axi.legend(title="map size")
    fig.tight_layout()
    img_title = "frozenlake_steps_and_rewards.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    # extract the best Q-values from the Q-table for each state
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    # get the corresponding best action for those Q-values
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    # map each action to an arrow so we can visualize it
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:  # Assign an arrow only if a minimal Q-value has been learned as best action otherwise since 0 is a direction, it also gets mapped on the tiles where it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def plot_q_values_map(qtable, env, map_size):
    """With the following function, we’ll plot on the left the last frame of the simulation.
    If the agent learned a good policy to solve the task,
    we expect to see it on the tile of the treasure in the last frame of the video.
    On the right we’ll plot the policy the agent has learned.
    Each arrow will represent the best action to choose for each tile/state."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def plot_states_actions_distribution(states, actions, map_size):
    """Plot the distributions of states and actions."""
    labels = {"←": 0, "↓": 1, "→": 2, "↑": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    #map_sizes = [4, 7, 10, 13]
    map_sizes = [7]
    res_all = pd.DataFrame()
    st_all = pd.DataFrame()

    for map_size in map_sizes:
        current_map = generate_random_map(
                size=map_size, p=params.prob_frozen, seed=params.seed
            )
        env = gym.make(
            "FrozenLake-v1",
            is_slippery=params.is_slippery,
            #render_mode="rgb_array",
            render_mode="human",  # to see the AI playing!
            desc=current_map
        )
        params = params._replace(action_size=env.action_space.n)
        params = params._replace(state_size=env.observation_space.n)
        env.action_space.seed(
            params.seed
        )

        learner = QLearningAgent(
            lr=params.lr,
            gamma=params.gamma,
            state_s=params.state_size,
            action_s=params.action_size,
        )
        explorer = EpsilonGreedy(
            e=params.e,
        )

        for ep in tqdm(
                range(10), desc=f"Episodes", leave=False
        ):
            state, _ = env.reset(seed=params.seed)
            done = False
            while not done:
                action = explorer.choose_action(
                    action_s=env.action_space, state=state, qtable=learner.qtable
                )
                next_s, _, term, trunc, _ = env.step(action)
                done = term or trunc
                state = next_s

        env.close()

        env = gym.make(
            "FrozenLake-v1",
            is_slippery=params.is_slippery,
            render_mode="rgb_array",
            #render_mode="human",  # to see the AI playing!
            desc=current_map
        )
        env.action_space.seed(
            params.seed
        )

        print(f"\nMap size: {map_size}x{map_size}")
        rewards, steps, episodes, qtables, all_states, all_actions = run_env()

        # Save the results in dataframes
        res, st = to_pandas(episodes, params, rewards, steps, map_size)
        res_all = pd.concat([res_all, res])
        st_all = pd.concat([st_all, st])
        qtable = qtables.mean(axis=0)  # Average the Q-table between runs

        #plot_states_actions_distribution(
        #    states=all_states, actions=all_actions, map_size=map_size
        #)  # Sanity check
        plot_q_values_map(qtable, env, map_size)

        env.close()

    #plot_steps_and_rewards(res_all, st_all)

    ################################
    #### SEE WHAT AGENT LEARNT #####
    ################################

    env = gym.make(
        "FrozenLake-v1",
        is_slippery=params.is_slippery,
        #render_mode="rgb_array",
        render_mode="human",  # to see the AI playing!
        desc=current_map
    )
    env.action_space.seed(
        params.seed
    )

    for ep in tqdm(
            range(10), desc=f"Episodes", leave=False
        ):
        state, _ = env.reset(seed=params.seed)
        done = False
        while not done:
            action = explorer.choose_action(
                action_s=env.action_space, state=state, qtable=learner.qtable
            )
            next_s, _, term, trunc, _ = env.step(action)
            done = term or trunc
            state = next_s

    env.close()
