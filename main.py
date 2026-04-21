import chess
import chess.pgn
import torch
import time

from ml.model import NNUE
from engine.search import ChessSearch

board = chess.Board()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NNUE()
# checkpoint = torch.load("ml/model_weights.pth")

# model.load_state_dict(checkpoint["model_state_dict"])
model.load_state_dict(torch.load("ml/model_weights.pth", map_location=device))
# model.load_state_dict(torch.load("ml/nnue_checkpoint.pt"))

engine = ChessSearch(model=model)

heurisitic_engine = ChessSearch(model=None)

pgn_game = chess.pgn.Game() # root
pgn_game.headers["White"] = "NNUE Engine"
pgn_game.headers["Black"] = "Heuristic Engine"

curr_node = pgn_game


while not board.is_game_over():

    print(board)
    print()

    if board.turn == chess.WHITE:
        start_time = time.perf_counter()
        ai_move = engine.find_best_move(board, depth=5)
        end_time = time.perf_counter()
        print(f"Engine plays: {ai_move} | time: {end_time - start_time}s")
        board.push(ai_move)
        curr_node = curr_node.add_variation(ai_move)

    else:
        ai_move = heurisitic_engine.find_best_move(board, depth=3)
        print("Heuristic Engine plays:", ai_move)
    
    if board.is_capture(ai_move):
        engine.clear_cache()
        heurisitic_engine.clear_cache()

    board.push(ai_move)
    curr_node = curr_node.add_variation(ai_move)

    engine.register_board(board)
    heurisitic_engine.register_board(board)

print("Game Over:", board.result())
pgn_game.headers["Result"] = board.result()
with open("./game.pgn", "w") as f:
    print(pgn_game, file=f)