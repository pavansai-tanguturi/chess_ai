Here’s a README.md for your Chess AI Reinforcement Learning project:

⸻

Chess AI - Reinforcement Learning (PPO)

This project trains a Chess AI using Reinforcement Learning (RL) with the Proximal Policy Optimization (PPO) algorithm in Stable-Baselines3. The AI learns to play chess by training against itself and can be deployed as an API.

Features
	•	Train AI using PPO algorithm
	•	Self-play training for White & Black agents
	•	Play AI vs AI games
	•	Deploy AI via FastAPI on Railway.app

⸻

1. Setup

1.1. Install Dependencies

pip install stable-baselines3 chess gym-chess numpy torch fastapi uvicorn

1.2. Clone the Repository

git clone https://github.com/pavansai-tanguturi/chess_ai.git
cd chess_ai



⸻

2. Train the Chess AI

2.1. Train White AI

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ChessEnv import ChessEnv

env = DummyVecEnv([lambda: ChessEnv()])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = PPO("MlpPolicy", env, verbose=1, device="cuda")  # Use GPU
model.learn(total_timesteps=10000)

model.save("chess_white_agent.zip")
env.save("chess_white_env.pkl")
env.close()

2.2. Train Black AI

black_env = DummyVecEnv([lambda: ChessEnv()])
black_model = PPO("MlpPolicy", black_env, verbose=1, device="cuda")
black_model.learn(total_timesteps=10000)

black_model.save("chess_black_agent.zip")
black_env.save("chess_black_env.pkl")
black_env.close()



⸻

3. Playing AI vs AI

from stable_baselines3 import PPO
from ChessEnv import ChessEnv

white_model = PPO.load("chess_white_agent.zip")
black_model = PPO.load("chess_black_agent.zip")

env = ChessEnv()
obs = env.reset()

for _ in range(100):
    action_white, _ = white_model.predict(obs)
    obs, _, done, _ = env.step(action_white)
    if done:
        break

    action_black, _ = black_model.predict(obs)
    obs, _, done, _ = env.step(action_black)
    if done:
        break

env.render()  # Show AI playing visually



⸻

4. Deploy as API

4.1. Create app.py

from fastapi import FastAPI
from ChessEnv import ChessEnv
from stable_baselines3 import PPO

app = FastAPI()
model = PPO.load("chess_white_agent.zip")
env = ChessEnv()

@app.get("/move")
def get_ai_move():
    obs = env.reset()
    action, _ = model.predict(obs)
    obs, _, done, _ = env.step(action)
    return {"move": action.tolist()}

4.2. Run API

uvicorn app:app --host 0.0.0.0 --port 8000



⸻

5. Deploy to Railway
	1.	Push the project to GitHub

git add .
git commit -m "Initial commit"
git push origin main


	2.	Go to Railway.app
	3.	Create a new service → GitHub Repo → Deploy

⸻

6. Troubleshooting

Issue	Solution
403 GitHub Error	Use a GitHub token instead of a password.
Colab session disconnects	Re-run GitHub setup & re-authenticate.
Model training slow	Use device="cuda" to enable GPU training.



⸻

7. Credits
	•	Author: pavansai-tanguturi
	•	Libraries Used: Stable-Baselines3, Gym-Chess, FastAPI

⸻


Here are some common errors you might face in your Chess AI reinforcement learning project and how to fix them:

⸻

1. GitHub Push Error: 403 Forbidden

Error:

fatal: unable to access 'https://github.com/.../chess_ai.git/': The requested URL returned error: 403

Cause:
	•	You’re using username & password instead of a GitHub token.
	•	You may not have write access to the repository.

Solution:
	•	Use GitHub Personal Access Token (PAT) instead of a password:

git remote set-url origin https://<TOKEN>@github.com/pavansai-tanguturi/chess_ai.git


	•	Or use SSH authentication:

git remote set-url origin git@github.com:pavansai-tanguturi/chess_ai.git



⸻

2. git push Rejected (fetch first required)

Error:

! [rejected]        main -> main (fetch first)

Cause:
	•	The remote repository has new commits that your local branch doesn’t have.

Solution:

git pull origin main --rebase
git push origin main

(If conflicts appear, resolve them before pushing.)

⸻

3. Colab Disconnecting (Session Expired)

Error:
	•	Training stops unexpectedly.
	•	Files disappear after session restart.

Solution:
	•	Save models periodically to Google Drive:

from google.colab import drive
drive.mount('/content/drive')
model.save("/content/drive/MyDrive/chess_ai_model.zip")


	•	Re-authenticate GitHub after reconnecting:

!git config --global user.email "your-email@example.com"
!git config --global user.name "your-username"
!git clone https://<TOKEN>@github.com/pavansai-tanguturi/chess_ai.git



⸻

4. GPU Not Being Used (device=cpu by default)

Error:
	•	Training is too slow.

Solution:
	•	Ensure GPU is enabled in Colab:
Runtime → Change runtime type → GPU
	•	Set device=“cuda” in PPO model:

model = PPO("MlpPolicy", env, verbose=1, device="cuda")



⸻

5. ModuleNotFoundError for stable_baselines3 or chess

Error:

ModuleNotFoundError: No module named 'stable_baselines3'

Solution:
	•	Install missing dependencies:

pip install stable-baselines3 chess gym-chess numpy torch



⸻

6. API Not Working on Railway (Error 502 Bad Gateway)

Cause:
	•	FastAPI server is not running properly.
	•	The port is incorrect.

Solution:
	•	Use 0.0.0.0 as host and set the correct port:

uvicorn app:app --host 0.0.0.0 --port 8000


	•	Add Procfile for Railway:

web: uvicorn app:app --host 0.0.0.0 --port $PORT



⸻

7. Chess AI Plays Illegal Moves

Cause:
	•	The AI doesn’t learn enough valid moves.
	•	The reward function in ChessEnv.py is not well-defined.

Solution:
	•	Train longer (total_timesteps=200000).
	•	Modify the reward function:

def reward_function(self):
    if self.done:
        return 1 if self.winner == self.current_player else -1
    return 0.01  # Small reward for valid moves



⸻
  
