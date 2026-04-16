import chess
from engine.eval import evaluate

INF = 999999

def minimax(board, depth, alpha, beta, maximizing):

    if depth == 0 or board.is_game_over():
        return evaluate(board), None

    best_move = None

    moves = list(board.legal_moves)

    if maximizing:
        best_score = -INF

        for move in moves:
            board.push(move)
            score, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)
            if beta <= alpha:
                break

        return best_score, best_move

    else:
        best_score = INF

        for move in moves:
            board.push(move)
            score, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()

            if score < best_score:
                best_score = score
                best_move = move

            beta = min(beta, score)
            if beta <= alpha:
                break

        return best_score, best_move


def find_best_move(board, depth=3):
    maximizing = board.turn == chess.WHITE
    _, move = minimax(board, depth, -INF, INF, maximizing)
    return move