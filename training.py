import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ChessEnv import ChessEnv
from stable_baselines3.common.utils import set_random_seed

# Set a random seed for reproducibility
set_random_seed(1)

# ✅ Function to create a properly wrapped environment
def make_env():
    env = DummyVecEnv([lambda: ChessEnv()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)  # Normalize env
    return env

# ✅ Use GPU for acceleration
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_float32_matmul_precision("high")  # Optimized matrix calculations

# ✅ Create a shared environment for both models
shared_env = make_env()

# ✅ Train White Model
white_model = PPO(
    "MlpPolicy",
    shared_env,  # Use shared env
    verbose=1,
    device=device,
    batch_size=512,
    n_steps=4096,  # Increased steps for faster training
    learning_rate=2.5e-4
)
white_model.learn(total_timesteps=50000)
white_model.save("chess_white_agent.zip")

# ✅ Train Black Model using the same env
black_model = PPO(
    "MlpPolicy",
    shared_env,  # Use the same env
    verbose=1,
    device=device,
    batch_size=512,
    n_steps=4096,  # Increased steps for faster training
    learning_rate=2.5e-4
)
black_model.learn(total_timesteps=50000)
black_model.save("chess_black_agent.zip")

# ✅ Save the shared environment normalization stats (Only One Save!)
shared_env.save("chess_env.pkl")
shared_env.close()  # Close the environment