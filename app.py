
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import chess
import chess.engine
from jinja2 import Template

app = FastAPI()

# Load Stockfish AI
ENGINE_PATH = "/usr/games/stockfish"  # Change if needed
engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

# HTML Template for Chess Board (using chessboard.js)
HTML_TEMPLATE = Template("""
<!DOCTYPE html>
<html lang="en">
<head>
    <title>AI Chess Game</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard.min.css">
    <style>
        #board { width: 400px; margin: 20px auto; }
        body { text-align: center; font-family: Arial, sans-serif; }
    </style>
</head>
<body>
    <h2>AI vs AI Chess</h2>
    <div id="board"></div>
    <script>
        var board = Chessboard('board', { position: 'start' });

        async function playAI() {
            let response = await fetch('/move');
            let data = await response.json();
            board.position(data.fen);
            setTimeout(playAI, 1000); // AI makes the next move every second
        }

        playAI();
    </script>
</body>
</html>
""")

# Chess Game State
board = chess.Board()

@app.get("/", response_class=HTMLResponse)
async def chess_ui():
    """ Serve the chessboard UI """
    return HTML_TEMPLATE.render()

@app.get("/move")
async def ai_move():
    """ AI makes a move and returns updated board """
    global board
    if board.is_game_over():
        return {"fen": board.fen(), "status": "Game Over"}

    result = engine.play(board, chess.engine.Limit(time=0.1))
    board.push(result.move)

    return {"fen": board.fen(), "status": "In Progress"}
