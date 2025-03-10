import chess
import chess.engine
import chess.svg
import gym
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from IPython.display import display, clear_output, SVG
import cloudpickle
import dill

class ChessEnv(Env):
    def __init__(self, stockfish_path="/usr/games/stockfish"):
        super().__init__()
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        # More realistic action space
        self.action_space = Discrete(4672)

        # Efficient observation space using uint8
        self.observation_space = Box(low=0, high=1, shape=(8, 8, 12), dtype=np.uint8)

    def reset(self):
        self.board.reset()
        return self.get_state()

    def get_state(self):
        """Converts the chess board into a 3D NumPy array for AI training."""
        state = np.zeros((8, 8, 12), dtype=np.uint8)
        for square, piece in self.board.piece_map().items():
            row, col = divmod(square, 8)
            state[row, col, piece.piece_type - 1] = 1  # One-hot encoding
        return state

    def step(self, action):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return self.get_state(), -1, True, {}

        # Execute player's move
        move = legal_moves[action % len(legal_moves)]
        self.board.push(move)

        # Check game status
        if self.board.is_checkmate():
            return self.get_state(), 1000, True, {}  # Player wins

        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return self.get_state(), 0, True, {}  # Draw

        # Stockfish moves
        result = self.engine.play(self.board, chess.engine.Limit(time=0.1))
        self.board.push(result.move)

        # Check result after AI move
        done = self.board.is_game_over()
        if done:
            result = self.board.result()
            if result == "0-1":  # AI wins
                reward = -1000
            elif result == "1-0":  # Player wins
                reward = 1000
            else:  # Draw
                reward = 0
        else:
            info = self.engine.analyse(self.board, chess.engine.Limit(depth=10))
            reward = info["score"].relative.score(mate_score=10000) / 100.0  # Evaluation-based rewards

        return self.get_state(), reward, done, {}

    def render(self, mode="human"):
        """Visualizes the chessboard in a Jupyter Notebook."""
        clear_output(wait=True)
        display(SVG(chess.svg.board(board=self.board, size=350)))

    def close(self):
        self.engine.quit()
