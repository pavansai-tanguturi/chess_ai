import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ChessEnv import ChessEnv
from stable_baselines3.common.utils import set_random_seed

# Set a random seed for reproducibility
set_random_seed(1)

# Function to create a properly wrapped environment
def make_env():
    env = DummyVecEnv([lambda: ChessEnv()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)  # Normalize env
    return env

# ✅ Use GPU with Torch Compile for Acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")  # Improves matrix calculations
torch.compile(PPO, backend="inductor")  # Optimize model execution

# ✅ Create Training Environment
white_env = make_env()
white_model = PPO(
    "MlpPolicy",  # Change to MlpPolicy for non-image inputs
    white_env,
    verbose=1,
    device=device,
    batch_size=512,
    n_steps=512,
    learning_rate=2.5e-4
)

# ✅ Train White Model Faster
white_model.learn(total_timesteps=15000)
white_model.save("chess_white_agent (2).zip")
white_env.save("chess_white_env.pkl")
white_env.close()

# ✅ Train Black Model Faster
black_env = make_env()
black_model = PPO(
    "MlpPolicy",  # Change to MlpPolicy for non-image inputs
    black_env,
    verbose=1,
    device=device,
    batch_size=512,
    n_steps=512,
    learning_rate=2.5e-4
)

black_model.learn(total_timesteps=10000)
black_model.save("chess_black_agent.zip")
black_env.save("chess_black_env.pkl")
black_env.close()