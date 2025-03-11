from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import chess
import chess.svg
from gameplay import load_agent, ChessEnv
from jinja2 import Template

app = FastAPI()

# Load AI agents
white_model, white_env = load_agent("chess_white_agent.zip", "chess_white_env.pkl")
black_model, black_env = load_agent("chess_black_agent.zip", "chess_black_env.pkl")

# Chess Board UI Template (uses chessboard.js)
HTML_TEMPLATE = Template("""
<!DOCTYPE html>
<html lang="en">
<head>
    <title>AI Chess Game</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
    <style>
        #board { width: 400px; margin: 20px auto; }
        body { text-align: center; font-family: Arial, sans-serif; }
    </style>
</head>
<body>
    <h2>AI vs AI Chess</h2>
    <div id="board"></div>
    <p id="status"></p>
    <script>
        var board = Chessboard('board', { position: 'start' });
        var game = new Chess();

        async function playAI() {
            let response = await fetch('/move');
            let data = await response.json();
            
            if (data.status === "Game Over") {
                document.getElementById("status").innerText = "Game Over: " + data.result;
            } else {
                game.load(data.fen);
                board.position(data.fen);
                setTimeout(playAI, 1000);
            }
        }

        playAI();
    </script>
</body>
</html>
""")

@app.get("/", response_class=HTMLResponse)
async def chess_ui():
    """ Serve the chessboard UI """
    return HTML_TEMPLATE.render()

@app.get("/move")
async def ai_move():
    """ AI makes a move and updates board """
    game_env = ChessEnv()
    obs, done = game_env.reset(), False

    if game_env.board.is_game_over():
        result = "Draw" if game_env.board.is_stalemate() else ("Black Wins" if game_env.board.turn else "White Wins")
        return {"fen": game_env.board.fen(), "status": "Game Over", "result": result}

    for model, env in [(white_model, white_env), (black_model, black_env)]:
        obs = env.normalize_obs(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = game_env.step(action)

        if done:
            break

    return {"fen": game_env.board.fen(), "status": "In Progress"}