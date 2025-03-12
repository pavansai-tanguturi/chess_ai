https://github.com/user-attachments/assets/97c909f1-3c62-4790-9a6f-b6f2f39dc1b5

# Chess AI - Reinforcement Learning with PPO  

This project is a **Chess AI** trained using **Reinforcement Learning (PPO Algorithm)** with **Stable-Baselines3**.  
It includes a **Flask-based web interface** where you can play or watch AI chess games.  

---

## 🚀 Features  

✅ **Train a Chess AI** using **Proximal Policy Optimization (PPO)**  
✅ **AI vs AI Chess Matches** using a trained model  
✅ **Interactive Web UI** to visualize chess games  
✅ **Flask API** to interact with the AI  

---

## 🛠️ How It Works  

1️⃣ **Training the AI** (`Training.py`)  
   - The AI is trained with **self-play**, where it learns by playing against itself.  
   - It uses the `ChessEnv.py` environment to make **legal moves** and improve its strategy.  
   - The trained model is saved for later use.  

2️⃣ **Playing a Chess Game** (`gameplay.py`)  
   - Loads the trained **AI model**.  
   - Runs an **AI vs AI** match step by step.  
   - Displays the moves on the **chessboard**.  

3️⃣ **Web Interface (`app.py` + `index.html`)**  
   - A Flask web server is used to host a **Chess UI**.  
   - The AI moves are visualized on the chessboard in a **browser**.  
   - Users can watch the game unfold automatically.  

---

## 📌 Installation  

### 1️⃣ Clone the Repository  
```bash
git clone <repo-url>
cd chess_ai



⸻

2️⃣ Set Up a Virtual Environment

🔹 Option 1: Using venv (Recommended)

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (cmd/Powershell)
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt



⸻

🔹 Option 2: Using conda

# Create a new Conda environment
conda create --name chess_ai python=3.9

# Activate the environment
conda activate chess_ai

# Install dependencies
pip install -r requirements.txt



⸻

3️⃣ Running the Chess AI Web App

python app.py

🎯 Open the web browser and go to http://127.0.0.1:5000

⸻

🎮 API Endpoints

Endpoint	Method	Description
/init_game	POST	Initializes a new game
/generate_game	POST	Runs AI vs AI game
/get_move/<index>	GET	Fetches a move from the game



⸻

📊 Training the AI

To train the AI, run:

python Training.py

This will generate a trained model that can be used for playing games.

⸻

🎭 Web Interface

Once the server is running, you will see:
🟢 Chessboard UI showing AI moves
🟢 Move History on the right
🟢 Game Playback as AI plays against itself

⸻

🏆 Future Improvements

✅ Add support for human vs AI gameplay
✅ Improve AI training efficiency
✅ Add difficulty levels for AI

⸻

🚀 Developed using Python, Flask, Stable-Baselines3, and Gym!

---

### 🔥 **What’s New in This README?**  
✔ **Explains how the Chess AI works** (Training, Playing, Web UI)  
✔ **Step-by-step guide for installation** (venv + conda)  
✔ **API documentation included**  
✔ **Web Interface details**  
