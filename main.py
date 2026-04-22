import chess
import chess.pgn
import torch
import time

from ml.model import NNUE
from engine.search import ChessSearch
from engine.eval import ChessModelEvaluator


FEN = "r1bqk1nr/pppp1ppp/2n5/2b1p1N1/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 1"
board = chess.Board(FEN)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NNUE()
checkpoint = torch.load("ml/nnue_orig.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

evaluator = ChessModelEvaluator(
    model=model,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

engine = ChessSearch(model=evaluator)
heurisitic_engine = ChessSearch()


pgn_game = chess.pgn.Game()
pgn_game.headers["Black"] = "NNUE Engine"
pgn_game.headers["White"] = "Heuristic Engine"
pgn_game.headers["FEN"] = FEN
pgn_game.headers["SetUp"] = "1"

curr_node = pgn_game

while not board.is_game_over():

    print(board)
    print()

    if board.turn == chess.BLACK:
        start_time = time.perf_counter()
        ai_move = engine.find_best_move(board, depth=5)
        end_time = time.perf_counter()
        print(f"NNUE Engine plays: {ai_move} | time: {end_time - start_time}s")

    else:
        start_time = time.perf_counter()
        ai_move = heurisitic_engine.find_best_move(board, depth=3)
        end_time = time.perf_counter()
        print(f"Heuristic Engine plays: {ai_move} | time: {end_time - start_time}s")

    if board.is_capture(ai_move):
        engine.clear_cache()
        heurisitic_engine.clear_cache()

    board.push(ai_move)
    curr_node = curr_node.add_variation(ai_move)

    engine.register_board(board)
    heurisitic_engine.register_board(board)

    with open("./game.pgn", "w") as f:
        print(pgn_game, file=f)

print("Game Over:", board.result())
pgn_game.headers["Result"] = board.result()

with open("./game.pgn", "w") as f:
    print(pgn_game, file=f)