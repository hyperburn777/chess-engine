import chess
from engine.search import find_best_move

board = chess.Board()

while not board.is_game_over():

    print(board)
    print()

    if board.turn == chess.WHITE:
        move = input("Your move: ")
        board.push_san(move)

    else:
        ai_move = find_best_move(board, depth=3)
        print("Engine plays:", ai_move)
        board.push(ai_move)

print("Game Over:", board.result())