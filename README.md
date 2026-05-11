# chess-engine
Developing a chess engine for machine learning project

Taking inspiration from:
- Stockfish
  - https://github.com/official-stockfish/Stockfish

- LCZero
  - https://github.com/LeelaChessZero

To run this project
- first run 'pip install -r requirements.txt'
- the dataset can be found at https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized
- to train the model, run 'train.ipynb', which should save the model at ml/nnue_checkpoints/chess_model_final.pt
- to run the game, 'python main.py' where moves are made like e2e4
- make sure to update the model used at torch.load to that which has been trained

To run the engine on lichess
- update the functions for 'uci.py'