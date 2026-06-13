import pickle
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repo_paths import FROZEN_LAKE_Q_SARSA, FROZEN_LAKE_RESULTS
from smoke_config import TABULAR_EPISODES, smoke_mode
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

RESULTS_DIR = FROZEN_LAKE_RESULTS
Q_VALUES_PATH = FROZEN_LAKE_Q_SARSA


class FrozenLakeAgent():
    def __init__(
        self, 
        env: gym.Env,
        epsilon: float, 
        epsilon_decay: float,
        learning_rate: float,
        discount_factor: float = 0.95) -> None:

        self.env = env
        self.lr = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.reward_history = []
        self.steps_history = []

        self.training_error = []
        self.total_steps = []

    def get_action(self, obs: int, training: bool = True) -> int:
        prob = np.random.random()

        if training and prob < self.epsilon:
            return self.env.action_space.sample()

        q = self.q_values[obs]
        best_actions = np.flatnonzero(np.isclose(q, q.max()))
        return int(np.random.choice(best_actions))


    def update_state(self, obs, action, reward, terminated, next_obs, next_action):      
        future_q = (not terminated) * self.q_values[next_obs][next_action]
        target = reward + self.discount_factor * future_q
        temporal_difference = target - self.q_values[obs][action]

        self.q_values[obs][action] = self.q_values[obs][action] + self.lr * temporal_difference



    def run_episode(self, training: bool = True):
        obs, info = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        action = self.get_action(obs, training=training)   # Key for SARSA: sample the first action before the loop so that
        # the next action drawn at the end of the loop becomes the current action at the start of the next iteration.
        
        while not done:            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            steps += 1
            total_reward += reward
            next_action = self.get_action(next_obs, training=training)

            if training:
                self.update_state(obs, action, reward, terminated, next_obs, next_action)

            obs = next_obs
            action = next_action

        if training:
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        
        return total_reward, steps

    def train(self, n_episodes: int = 1000):
        for i in tqdm(range(n_episodes)):
            reward, steps = self.run_episode()
            self.reward_history.append(reward)
            self.steps_history.append(steps)

            if (i + 1) % 100 == 0:
                avg_reward = np.mean(self.reward_history[-100:])
                avg_steps = np.mean(self.steps_history[-100:])
                print(f"Episode {i+1}/{n_episodes} | Avg Reward (last 100): {avg_reward:.2f} | Avg Steps (last 100): {avg_steps:.1f} | Epsilon: {self.epsilon:.3f}")

# 4x4 Frozen Lake board layout: SFFF / FHFH / FFFH / HFFG
# States 0-15: row = state//4, column = state%4
FROZEN_LAKE_4x4_TILES = [
    'S', 'F', 'F', 'F',  # row 0
    'F', 'H', 'F', 'H',  # row 1
    'F', 'F', 'F', 'H',  # row 2
    'H', 'F', 'F', 'G',  # row 3
]
# Actions: 0=Left, 1=Down, 2=Right, 3=Up


def plot_q_table_grid(agent, filename: str = 'Q_grid_SARSA.png'):
    """
    Visualization of the 4x4 board with Q-values arranged by movement direction.
    In each cell: top=Up, bottom=Down, left=Left, right=Right.
    The 2 highest values = green, the rest = red.
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    n_rows, n_cols = 4, 4
    cell_size = 1.0

    tile_colors = {'S': '#87CEEB', 'F': '#E0F7FA', 'H': '#424242', 'G': '#81C784'}

    for row in range(n_rows):
        for col in range(n_cols):
            state = row * n_cols + col
            tile = FROZEN_LAKE_4x4_TILES[state]
            q_vals = agent.q_values[state]

            x_center = col * cell_size + cell_size / 2
            y_center = (n_rows - 1 - row) * cell_size + cell_size / 2

            rect_x = col * cell_size
            rect_y = (n_rows - 1 - row) * cell_size
            rect = plt.Rectangle((rect_x, rect_y), cell_size, cell_size,
                                 facecolor=tile_colors[tile], edgecolor='black', linewidth=2)
            ax.add_patch(rect)

            ax.text(x_center, y_center, tile, fontsize=24, fontweight='bold',
                    ha='center', va='center')

            top2_indices = set(np.argsort(q_vals)[::-1][:2])
            color_best = '#2E7D32'
            color_rest = '#C62828'   # red

            offset_vert = 0.35
            offset_horiz = 0.22
            fontsize = 27
            fmt = '.2f'

            ax.text(x_center - offset_horiz, y_center, f'{q_vals[0]:{fmt}}',
                    fontsize=fontsize, ha='right', va='center',
                    color=color_best if 0 in top2_indices else color_rest)
            ax.text(x_center, y_center - offset_vert, f'{q_vals[1]:{fmt}}',
                    fontsize=fontsize, ha='center', va='top',
                    color=color_best if 1 in top2_indices else color_rest)
            ax.text(x_center + offset_horiz, y_center, f'{q_vals[2]:{fmt}}',
                    fontsize=fontsize, ha='left', va='center',
                    color=color_best if 2 in top2_indices else color_rest)
            ax.text(x_center, y_center + offset_vert, f'{q_vals[3]:{fmt}}',
                    fontsize=fontsize, ha='center', va='bottom',
                    color=color_best if 3 in top2_indices else color_rest)

    ax.set_xlim(0, n_cols * cell_size)
    ax.set_ylim(0, n_rows * cell_size)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Frozen Lake 4x4 - Q values (SARSA)\nLeft=Left | Down=Down | Right=Right | Up=Up',
                 fontsize=12)

    plt.tight_layout()
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Board visualization saved as {path}")
    plt.close()


def print_q_table(agent):
    print("\n--- Q-Table (State | Left | Down | Right | Up) ---")
    header = f"{'State':<6} | {'Left (0)':<8} | {'Down (1)':<8} | {'Right (2)':<8} | {'Up (3)':<8}"
    print(header)
    print("-" * len(header))
    
    for state in range(agent.env.observation_space.n):
        q_vals = agent.q_values[state]
        print(f"{state:<6} | {q_vals[0]:8.4f} | {q_vals[1]:8.4f} | {q_vals[2]:8.4f} | {q_vals[3]:8.4f}")


def plot_results(rewards, steps):
    """
    Plots the rolling average of rewards and steps to show learning progress.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # Calculate rolling average for smoother visualization
    rolling_avg_reward = np.convolve(rewards, np.ones(100)/100, mode='valid')
    ax1.plot(rolling_avg_reward)
    ax1.set_title("Learning Progress (Rolling Average Reward)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Reward (last 100)")
    ax1.grid(True)

    rolling_avg_steps = np.convolve(steps, np.ones(100)/100, mode='valid')
    ax2.plot(rolling_avg_steps, color='orange')
    ax2.set_title("Steps per Episode (Rolling Average)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Average Steps (last 100)")
    ax2.grid(True)

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "learning_curve_SARSA.png"
    plt.savefig(path)
    print(f"Plot saved as {path}")
    plt.close()


def save_q_values(agent, path: Path = Q_VALUES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(dict(agent.q_values), f)
    print(f"Q-values saved to {path}")


# --- Execution ---

if __name__ == "__main__":
    n_episodes = TABULAR_EPISODES if smoke_mode() else 10_000
    if smoke_mode():
        print(f"Smoke mode: training {n_episodes} episodes")

    # 1. Initialize environment (without render for fast training)
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)

    # 2. Initialize Agent
    agent = FrozenLakeAgent(
        env=env,
        epsilon=1.0,
        epsilon_decay=0.999,  # Slow decay for better exploration
        learning_rate=0.1,
        discount_factor=0.9
    )

    # 3. Train
    agent.train(n_episodes=n_episodes)

    # 4. Plot statistics
    if not smoke_mode():
        plot_results(agent.reward_history, agent.steps_history)
        print_q_table(agent)
        plot_q_table_grid(agent, RESULTS_DIR / "Q_grid_SARSA.png")
    save_q_values(agent)
