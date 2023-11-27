import tensorflow as tf
import chess
import joblib
import numpy as np
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_file = './models/model1.pkl'

model = joblib.load(model_file)


def boardstate(fen):
    board = chess.Board(fen)
    fstr = str(fen)

    if board.has_kingside_castling_rights(chess.WHITE) == True:
        WCKI = 1
    else:
        WCKI = 0
    if board.has_queenside_castling_rights(chess.WHITE) == True:
        WCQ = 1
    else:
        WCQ = 0
    if board.is_check() == True:
        WCH = 1
    else:
        WCH = 0

    if board.has_kingside_castling_rights(chess.BLACK) == True:
        BCKI = 1
    else:
        BCKI = 0
    if board.has_queenside_castling_rights(chess.BLACK) == True:
        BCQ = 1
    else:
        BCQ = 0
    if board.was_into_check() == True:
        BCH = 1
    else:
        BCH = 0

    # f = [M, WCKI, WCQ, WCH, BCKI, BCQ, BCH]
    fw = [WCKI, WCQ, WCH]
    fb = [BCKI, BCQ, BCH]

    bstr = str(board)
    bstr = bstr.replace("p", "\ -1")
    bstr = bstr.replace("n", "\ -3")
    bstr = bstr.replace("b", "\ -4")
    bstr = bstr.replace("r", "\ -5")
    bstr = bstr.replace("q", "\ -9")
    bstr = bstr.replace("k", "\ -100")
    bstr = bstr.replace("P", "\ 1")
    bstr = bstr.replace("N", "\ 3")
    bstr = bstr.replace("B", "\ 4")
    bstr = bstr.replace("R", "\ 5")
    bstr = bstr.replace("Q", "\ 9")
    bstr = bstr.replace("K", "\ 100")
    bstr = bstr.replace(".", "\ 0")
    bstr = bstr.replace("\ ", ",")
    bstr = bstr.replace("'", " ")
    bstr = bstr.replace("\n", "")
    bstr = bstr.replace(" ", "")
    bstr = bstr[1:]
    bstr = eval(bstr)
    bstr = list(bstr)
    if "w" not in fstr:
        for i in range(len(bstr)):
            bstr[i] = bstr[i] * -1
        bstr.reverse()
        fs = fb
        fb = fw
        fw = fs

    BITBOARD = fw + fb + bstr

    return BITBOARD


def evaluate_position(model, fen):

    input2_columns = [0, 1, 2, 3, 4, 5]

    test_position = boardstate(fen.fen())

    test_position = pd.DataFrame(test_position).T

    inputboard = test_position.drop(
        columns=test_position.iloc[:, input2_columns])
    inputboard = np.array(inputboard)
    inputmeta = test_position.iloc[:, input2_columns]
    inputmeta = np.array(inputmeta)
    eval_score = model.predict([(inputboard), (inputmeta)])[0][0]
    return eval_score


def minimax(position, depth, maximizing_player, evaluate_position):
    if depth == 0 or position.is_game_over():
        return evaluate_position(model, position)

    if maximizing_player:
        max_eval = float('-inf')
        for move in position.legal_moves:
            position.push(move)
            eval_score = minimax(position, depth - 1, False, evaluate_position)
            position.pop()
            max_eval = max(max_eval, eval_score)
        return max_eval
    else:
        min_eval = float('inf')
        for move in position.legal_moves:
            position.push(move)
            eval_score = minimax(position, depth - 1, True, evaluate_position)
            position.pop()
            min_eval = min(min_eval, eval_score)
        return min_eval


def choose_best_move(position, depth):
    best_move = None
    best_eval = float('-inf')
    for move in position.legal_moves:
        position.push(move)
        eval_score = minimax(position, depth - 1, False, evaluate_position)
        position.pop()
        if eval_score > best_eval:
            best_eval = eval_score
            best_move = move
    return best_move
