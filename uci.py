import sys
import threading
import time

import chess
import chess.polyglot
import torch

from ml.model import NNUE
from engine.search import ChessSearch

ENGINE_NAME = "NNUE Engine"
AUTHOR_NAME = "ML Chess Engine Research"
DEFAULT_DEPTH = 3
DEFAULT_HASH_MB = 64
DEFAULT_THREADS = 1
DEFAULT_SYZYGY_PATH = ""
DEFAULT_UCI_SHOW_WDL = False
MOVE_OVERHEAD_SEC = 0.05
DEFAULT_MAX_THINK_SEC = 5
DEFAULT_TARGET_DEPTH = 5
HEARTBEAT_INTERVAL_SEC = 3.0


def eprint(*args):
    print(*args, file=sys.stderr, flush=True)


def load_engine():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = NNUE().to(device)
    checkpoint = torch.load(
        "ml/nnue_checkpoints/chess_model_final.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return ChessSearch(model=model)


class UCIEngine:
    def __init__(self):
        self.engine = load_engine()
        self.board = chess.Board()
        self.default_depth = DEFAULT_DEPTH
        self.hash_mb = DEFAULT_HASH_MB
        self.threads = DEFAULT_THREADS
        self.syzygy_path = DEFAULT_SYZYGY_PATH
        self.show_wdl = DEFAULT_UCI_SHOW_WDL
        self.max_think_sec = DEFAULT_MAX_THINK_SEC
        self.ponder = False

        self.stop_event = threading.Event()
        self.search_thread = None
        self.heartbeat_thread = None
        self.search_lock = threading.Lock()
        self.search_result = None
        self.search_announced = False

    def _reset_search_state(self):
        self.stop_event = threading.Event()
        self.search_result = None
        self.search_announced = False
        self.heartbeat_thread = None

    def _search_heartbeat(self):
        while not self.stop_event.wait(HEARTBEAT_INTERVAL_SEC):
            print("info string searching", flush=True)

    def _stop_active_search(self, wait=True):
        thread = self.search_thread
        if thread is None:
            return

        self.stop_event.set()
        if wait and thread.is_alive():
            thread.join(timeout=0.2)

        if self.search_result is not None and not self.search_announced:
            self._announce_bestmove(self.search_result)

        self.search_thread = None
        self.heartbeat_thread = None

    def _announce_bestmove(self, move):
        if move is None:
            print("bestmove 0000", flush=True)
        else:
            print(f"bestmove {move.uci()}", flush=True)
        self.search_announced = True

    def _set_board_from_position(self, tokens):
        if len(tokens) < 2:
            return chess.Board()

        index = 1
        if tokens[index] == "startpos":
            board = chess.Board()
            index += 1
        elif tokens[index] == "fen":
            fen_tokens = tokens[index + 1:index + 7]
            if len(fen_tokens) != 6:
                eprint("Invalid fen in position command")
                return chess.Board()
            try:
                board = chess.Board(" ".join(fen_tokens))
            except ValueError:
                eprint("Failed to parse fen in position command")
                return chess.Board()
            index += 7
        else:
            eprint(f"Unsupported position command: {' '.join(tokens)}")
            return chess.Board()

        if index < len(tokens) and tokens[index] == "moves":
            for move_uci in tokens[index + 1:]:
                try:
                    board.push_uci(move_uci)
                except ValueError:
                    eprint(f"Ignoring illegal move in position command: {move_uci}")
                    break

        return board

    def _seed_repetition_history(self, board):
        self.engine.move_cache.clear()
        temp_board = chess.Board()
        self.engine.move_cache.add(chess.polyglot.zobrist_hash(temp_board))

        for move in board.move_stack:
            temp_board.push(move)
            self.engine.move_cache.add(chess.polyglot.zobrist_hash(temp_board))

    def _parse_go(self, tokens):
        params = {}
        index = 1

        while index < len(tokens):
            key = tokens[index]

            if key in {"depth", "movetime", "wtime", "btime", "winc", "binc", "movestogo", "nodes"}:
                if index + 1 < len(tokens):
                    try:
                        params[key] = int(tokens[index + 1])
                    except ValueError:
                        pass
                    index += 2
                else:
                    index += 1
            elif key in {"ponder", "infinite"}:
                params[key] = True
                index += 1
            else:
                index += 1

        return params

    def _time_budget_seconds(self, go_params):
        if self.board.turn == chess.WHITE:
            remaining_ms = go_params.get("wtime", 0)
            increment_ms = go_params.get("winc", 0)
        else:
            remaining_ms = go_params.get("btime", 0)
            increment_ms = go_params.get("binc", 0)

        moves_to_go = max(1, go_params.get("movestogo", 30))
        budget_sec = (remaining_ms / 1000.0) / moves_to_go
        budget_sec += (increment_ms / 1000.0) * 0.8
        budget_sec -= MOVE_OVERHEAD_SEC
        return max(0.01, budget_sec)

    def _search_best_move(self, board, go_params):
        if go_params.get("infinite") or go_params.get("ponder"):
            best_move = None
            depth = 1
            while not self.stop_event.is_set():
                move = self.engine.find_best_move(board, depth=depth, stop_event=self.stop_event)
                if self.stop_event.is_set():
                    break
                if move is not None:
                    best_move = move
                depth += 1
            return best_move

        if "movetime" in go_params:
            requested = max(0.01, go_params["movetime"] / 1000.0 - MOVE_OVERHEAD_SEC)
            limit = min(requested, getattr(self, "max_think_sec", requested))
            return self.engine.find_best_move_tl(
                board,
                limit,
                stop_event=self.stop_event,
            )

        if "wtime" in go_params or "btime" in go_params:
            budget = self._time_budget_seconds(go_params)
            limit = min(budget, getattr(self, "max_think_sec", budget))
            return self.engine.find_best_move_tl(
                board,
                limit,
                stop_event=self.stop_event,
            )

        # If depth is explicitly specified, use depth-limited search
        if "depth" in go_params:
            depth = go_params["depth"]
            return self.engine.find_best_move_depth(board, depth, stop_event=self.stop_event)
        
        # Default: use engine's time-limited iterative deepening with transposition table
        limit = getattr(self, "max_think_sec", DEFAULT_MAX_THINK_SEC)
        return self.engine.find_best_move_tl(
            board,
            limit,
            stop_event=self.stop_event,
        )

    def _search_worker(self, board_snapshot, go_params):
        try:
            move = self._search_best_move(board_snapshot, go_params)

            with self.search_lock:
                self.search_result = move
                # Timed searches may set stop_event internally at the deadline;
                # if we found a move, we must still emit bestmove.
                should_announce = (move is not None) or (not self.stop_event.is_set())

            if should_announce:
                self._announce_bestmove(move)
        except Exception as exc:
            # Never fail silently in a UCI search thread; return a null move so
            # the GUI/bot doesn't hang waiting for bestmove.
            eprint(f"Search worker exception: {exc}")
            print(f"info string search error: {exc}", flush=True)
            with self.search_lock:
                self.search_result = None
            if not self.stop_event.is_set():
                self._announce_bestmove(None)
        finally:
            self.stop_event.set()

    def _start_search(self, go_params):
        self._stop_active_search(wait=True)
        self._reset_search_state()

        board_snapshot = self.board.copy(stack=True)
        # Pre-initialize engine state (accumulators, caches) so the first
        # iterative-deepening iteration returns quickly.
        try:
            self.engine.register_board(board_snapshot)
        except Exception:
            pass
        self.heartbeat_thread = threading.Thread(target=self._search_heartbeat, daemon=True)
        self.heartbeat_thread.start()
        self.search_thread = threading.Thread(
            target=self._search_worker,
            args=(board_snapshot, go_params),
            daemon=True,
        )
        self.search_thread.start()

    def _handle_setoption(self, tokens):
        if "name" not in tokens:
            return

        name_index = tokens.index("name") + 1
        value_index = tokens.index("value") if "value" in tokens else len(tokens)
        option_name = " ".join(tokens[name_index:value_index]).strip().lower()
        option_value = ""

        if "value" in tokens and value_index + 1 < len(tokens):
            option_value = " ".join(tokens[value_index + 1:]).strip()

        if option_name == "depth":
            try:
                self.default_depth = max(1, min(20, int(option_value)))
            except ValueError:
                pass
        elif option_name == "threads":
            try:
                # Currently informational only; search is single-threaded.
                self.threads = max(1, int(option_value))
            except ValueError:
                pass
        elif option_name == "syzygypath":
            # Stored for compatibility with UCI clients that always set tablebase paths.
            self.syzygy_path = option_value
        elif option_name == "uci_showwdl":
            self.show_wdl = option_value.lower() in {"true", "1", "on", "yes"}
        elif option_name == "maxthinktime":
            try:
                self.max_think_sec = max(1, int(option_value))
            except ValueError:
                pass
        elif option_name == "hash":
            try:
                self.hash_mb = max(1, int(option_value))
            except ValueError:
                pass
        elif option_name == "move overhead":
            try:
                global MOVE_OVERHEAD_SEC
                MOVE_OVERHEAD_SEC = max(0.0, int(option_value) / 1000.0)
            except ValueError:
                pass
        elif option_name == "ponder":
            self.ponder = option_value.lower() in {"true", "1", "on", "yes"}

    def handle_line(self, line):
        if not line:
            return

        tokens = line.split()
        if not tokens:
            return

        command = tokens[0]

        if command == "uci":
            print(f"id name {ENGINE_NAME}")
            print(f"id author {AUTHOR_NAME}")
            print(f"option name Depth type spin default {DEFAULT_DEPTH} min 1 max 20")
            print(f"option name Threads type spin default {DEFAULT_THREADS} min 1 max 512")
            print(f"option name Hash type spin default {DEFAULT_HASH_MB} min 1 max 4096")
            print(f"option name SyzygyPath type string default {DEFAULT_SYZYGY_PATH}")
            print(f"option name UCI_ShowWDL type check default {str(DEFAULT_UCI_SHOW_WDL).lower()}")
            print(f"option name MaxThinkTime type spin default {DEFAULT_MAX_THINK_SEC} min 1 max 300")
            print("option name Move Overhead type spin default 50 min 0 max 5000")
            print("option name Ponder type check default false")
            print("uciok", flush=True)

        elif command == "isready":
            print("readyok", flush=True)

        elif command == "setoption":
            self._handle_setoption(tokens)

        elif command == "ucinewgame":
            self._stop_active_search(wait=True)
            self.board = chess.Board()
            self._reset_search_state()

        elif command == "position":
            self._stop_active_search(wait=True)
            self.board = self._set_board_from_position(tokens)
            self._seed_repetition_history(self.board)

        elif command == "go":
            go_params = self._parse_go(tokens)
            self._start_search(go_params)

        elif command == "stop":
            self._stop_active_search(wait=True)

        elif command == "ponderhit":
            self.stop_event.clear()

        elif command == "quit":
            self._stop_active_search(wait=True)
            raise SystemExit

        elif command == "debug":
            eprint(f"DEBUG: {line}")

        else:
            return


def main():
    sys.stdout.reconfigure(line_buffering=True)
    uci = UCIEngine()

    try:
        while True:
            line = sys.stdin.readline()
            if line == "":
                break
            uci.handle_line(line.strip())
    except SystemExit:
        pass


if __name__ == "__main__":
    main()