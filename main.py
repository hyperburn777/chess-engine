import chess
import torch

from ml.model import NNUE
from engine.search import ChessSearch
from engine.eval import ChessModelEvaluator

board = chess.Board()

model = NNUE()
# model.load_state_dict(torch.load("ml/model_weights.pth"))
checkpoint = torch.load("ml/nnue_checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])

evaluator = ChessModelEvaluator(model=model, device="cuda" if torch.cuda.is_available() else "cpu")
engine = ChessSearch(model=evaluator)

heurisitic_engine = ChessSearch()

while not board.is_game_over():

    print(board)
    print()

    if board.turn == chess.WHITE:
        ai_move = engine.find_best_move(board, depth=3)
        print("Engine plays:", ai_move)
        board.push(ai_move)

    else:
        ai_move = heurisitic_engine.find_best_move(board, depth=3)
        print("Heuristic Engine plays:", ai_move)
        board.push(ai_move)

print("Game Over:", board.result())