from flask import Flask, render_template, request, jsonify
import chess
import chess.svg
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ChessEnv import ChessEnv

app = Flask(__name__)

# Global variables to store game state
env = None
white_model = None
black_model = None
current_turn = chess.WHITE  # Start with white
game_moves = []  # Store moves for step-by-step playback

def load_models():
    """Load AI models"""
    global white_model, black_model, env
    
    try:
        # Create a base environment for the game
        env = ChessEnv()
        
        # Load the trained models
        white_env = VecNormalize.load("chess_white_env.pkl", DummyVecEnv([ChessEnv]))
        white_env.training = False
        white_env.norm_obs = True
        white_model = PPO.load("chess_white_agent.zip", env=white_env)
        
        black_env = VecNormalize.load("chess_black_env.pkl", DummyVecEnv([ChessEnv]))
        black_env.training = False
        black_env.norm_obs = True
        black_model = PPO.load("chess_black_agent.zip", env=black_env)
        
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/init_game', methods=['POST'])
def init_game():
    """Initialize a new game"""
    global env, current_turn, game_moves
    
    if env is None:
        success = load_models()
        if not success:
            return jsonify({'error': 'Failed to load AI models'})
    
    # Reset the environment and game state
    obs = env.reset()
    current_turn = chess.WHITE
    game_moves = []  # Clear previous game moves
    
    # Get the SVG representation of the board
    board_svg = chess.svg.board(board=env.board, size=400)
    
    return jsonify({
        'board_svg': board_svg,
        'fen': env.board.fen(),
        'game_over': env.board.is_game_over(),
        'status': 'Game initialized. White to move.'
    })

@app.route('/generate_game', methods=['POST'])
def generate_game():
    """Generate a full AI vs AI game but don't visualize it yet"""
    global env, white_model, black_model, game_moves
    
    if env is None:
        success = load_models()
        if not success:
            return jsonify({'error': 'Failed to load AI models'})
    
    # Reset the environment
    obs = env.reset()
    game_moves = []
    
    try:
        # Play the game
        done = False
        while not done:
            for model, model_env_path, color in [
                (white_model, "chess_white_env.pkl", "White"),
                (black_model, "chess_black_env.pkl", "Black")
            ]:
                if done:
                    break
                
                # Load the environment
                model_env = VecNormalize.load(model_env_path, DummyVecEnv([ChessEnv]))
                model_env.training = False
                model_env.norm_obs = True
                
                # Get normalized observation
                normalized_obs = model_env.normalize_obs(obs)
                
                # Get action from model
                action, _ = model.predict(normalized_obs, deterministic=True)
                
                # Make the move
                board_before = env.board.copy()
                obs, reward, done, _ = env.step(action)
                
                # Record the move
                if len(env.board.move_stack) > len(board_before.move_stack):
                    last_move = str(env.board.move_stack[-1])
                    # Store move info
                    game_moves.append({
                        'color': color,
                        'move': last_move,
                        'fen': env.board.fen()
                    })
        
        # Return the number of moves
        return jsonify({
            'success': True,
            'total_moves': len(game_moves)
        })
    
    except Exception as e:
        return jsonify({'error': f'Game generation error: {str(e)}'})

@app.route('/get_move/<int:move_index>', methods=['GET'])
def get_move(move_index):
    """Get a specific move from the generated game"""
    global game_moves
    
    if move_index < 0 or move_index >= len(game_moves):
        return jsonify({'error': 'Invalid move index'})
    
    move = game_moves[move_index]
    
    # Create a temporary board to get the SVG for this position
    temp_board = chess.Board(move['fen'])
    
    # Get the last move to highlight
    if temp_board.move_stack:
        last_move = temp_board.move_stack[-1]
        board_svg = chess.svg.board(board=temp_board, size=400, lastmove=last_move)
    else:
        board_svg = chess.svg.board(board=temp_board, size=400)
    
    # Determine if this is the last move
    is_last_move = (move_index == len(game_moves) - 1)
    
    # Get game status for the last move
    status = ""
    if is_last_move:
        status = get_game_status(temp_board)
    else:
        status = f"{move['color']} moved {move['move']}"
    
    return jsonify({
        'board_svg': board_svg,
        'move': f"{move['color']}-{move['move']}",
        'is_last_move': is_last_move,
        'status': status
    })

def get_game_status(board=None):
    """Get the current game status as a string"""
    global env
    
    # Use provided board or default to the environment board
    if board is None:
        board = env.board
    
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"Checkmate! {winner} wins."
    elif board.is_stalemate():
        return "Game ended in stalemate."
    elif board.is_insufficient_material():
        return "Draw due to insufficient material."
    elif board.is_seventyfive_moves():
        return "Draw due to 75-move rule."
    elif board.is_fivefold_repetition():
        return "Draw due to fivefold repetition."
    elif board.is_check():
        return f"{'White' if board.turn == chess.WHITE else 'Black'} is in check."
    else:
        return f"{'White' if board.turn == chess.WHITE else 'Black'} to move."

if __name__ == '__main__':
    app.run(debug=True)