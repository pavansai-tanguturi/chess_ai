from fastapi import FastAPI
from pydantic import BaseModel
import chess
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ChessEnv import ChessEnv

app = FastAPI()

# Load AI models
def load_agent(model_path, env_path):
    env = VecNormalize.load(env_path, DummyVecEnv([ChessEnv]))
    env.training = False
    env.norm_obs = True
    return PPO.load(model_path, env=env), env

white_model, white_env = load_agent("chess_white_agent.zip", "chess_white_env.pkl")
black_model, black_env = load_agent("chess_black_agent.zip", "chess_black_env.pkl")

# Initialize game
game_env = ChessEnv()
obs, done = game_env.reset(), False

@app.get("/")
def read_root():
    return {"message": "AI Chess API is running"}

@app.get("/move")
async def ai_move():
    """ AI makes a move and returns updated board state """
    global obs, done

    if game_env.board.is_game_over():
        return {"fen": game_env.board.fen(), "status": "Game Over"}

    for model, env, color in [(white_model, white_env, "White"), (black_model, black_env, "Black")]:
        time.sleep(1)  # Simulate AI thinking time

        obs = env.normalize_obs(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = game_env.step(action)

        if done:
            break  # Stop if game ends

    return {"fen": game_env.board.fen(), "status": "In Progress"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)