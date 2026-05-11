import chess
import chess.pgn
import torch
import time

from ml.model import NNUE
from engine.search import ChessSearch

board = chess.Board()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = NNUE().to(device)
# checkpoint = torch.load("ml/model_weights.pth")
checkpoint = torch.load("ml/nnue_checkpoints/chess_model_final.pt", map_location=device, weights_only=True)

model.load_state_dict(checkpoint["model_state_dict"])
# model.load_state_dict(torch.load("ml/model_weights.pth", map_location=device))
# model.load_state_dict(torch.load("ml/nnue_checkpoint.pt"))

# evaluator = ChessModelEvaluator(model=model, device="cuda" if torch.cuda.is_available() else "cpu")
engine = ChessSearch(model=model)

heurisitic_engine = ChessSearch()

pgn_game = chess.pgn.Game() # root
pgn_game.headers["White"] = "NNUE Engine"
pgn_game.headers["Black"] = "Heuristic Engine"

curr_node = pgn_game


while not board.is_game_over():

    print(board)
    print()

    if board.turn == chess.WHITE:
        start_time = time.perf_counter()
        ai_move = engine.find_best_move(board, depth=4)
        # ai_move = engine.find_best_move_tl(board, 10)
        end_time = time.perf_counter()
        print(f"Engine plays: {ai_move} | time: {end_time - start_time}s")

    else:
        # ai_move = heurisitic_engine.find_best_move(board, depth=4)
        # ai_move = engine.find_best_move(board, depth=4)

        while True:
            move_str = input("Enter your move (e.g., e2e4): ")
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    print(f"You play: {move}")
                    break
                else:
                    print("That move isn't legal right now!")
            except:
                print("Invalid move format! Use UCI notation (e.g., e2e4)")
        ai_move = move

        # print("Heuristic Engine plays:", ai_move)
    
    if board.is_capture(ai_move):
        engine.clear_cache()
        heurisitic_engine.clear_cache()

    board.push(ai_move)
    curr_node = curr_node.add_variation(ai_move)

    engine.register_board(board)
    heurisitic_engine.register_board(board)

result = board.result()

# Map result to human-readable format based on who actually plays which color
white_player = pgn_game.headers["White"]
black_player = pgn_game.headers["Black"]

result_map = {
    "1-0": f"White wins ({white_player})",
    "0-1": f"Black wins ({black_player})",
    "1/2-1/2": "Draw"
}
result_text = result_map.get(result, result)

print("Game Over:", result_text)
pgn_game.headers["Result"] = result
with open("./game.pgn", "w") as f:
    print(pgn_game, file=f)