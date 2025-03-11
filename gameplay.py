import time
import chess
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ChessEnv import ChessEnv

# Load model and environment
def load_agent(model_path, env_path):
    env = VecNormalize.load(env_path, DummyVecEnv([ChessEnv]))
    env.training = False  # Ensure evaluation mode
    env.norm_obs = True   # Enable observation normalization
    return PPO.load(model_path, env=env), env

# Load AI agents
white_model, white_env = load_agent("chess_white_agent.zip", "chess_white_env.pkl")
black_model, black_env = load_agent("chess_black_agent.zip", "chess_black_env.pkl")

# Initialize game
game_env = ChessEnv()
obs, done = game_env.reset(), False

# Ensure White starts
if game_env.board.turn != chess.WHITE:
    print("⚠️ Error: White should start but Black has the turn!")

print("\n♟️ AI vs AI Chess Match Starts! ♟️\n")

# AI Game Loop
while not done:
    for model, env, color in [(white_model, white_env, "White"), (black_model, black_env, "Black")]:
        time.sleep(1)  # ⏳ Add a delay of 2 seconds between moves

        obs = env.normalize_obs(obs)
        action, _ = model.predict(obs, deterministic=True)

        print(f"{color} chose action: {action}")  # Debug chosen action
        obs, reward, done, info = game_env.step(action)

        print(f"{color} received reward: {reward}")  # Debug reward values
        print(f"\n{color} (AI) moves:")
        game_env.render()

        if done:
            break  # Stop if game ends

print("\n♟️ Game Over! ♟️")

# Determine winner
board = game_env.board
if board.is_checkmate():
    print(f"{'Black' if board.turn else 'White'} (AI) wins by checkmate!")
elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
    print("The game ended in a draw!")
else:
    print("Unable to determine the winner. Check ChessEnv logic.")