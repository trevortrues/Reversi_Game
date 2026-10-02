#Zijie Zhang, Sep.24/2023
#
#This is the only file you need to change. reversi_client.py calls choose_move() every time it is your turn.

import numpy as np
from reversi import reversi


def choose_move(board, turn, time_limit):
    """Pick the square to play and return it as (x, y).

    board      : 8x8 numpy array. 1 = white piece, -1 = black piece, 0 = empty.
                 board[x, y] is row x, column y, exactly as print(board) and the game window show it.
    turn       : 1 if you are playing white, -1 if you are playing black.
    time_limit : seconds you have to answer, or None if there is no limit.
                 Answer late, or with an illegal move, and you forfeit this turn.

    You are only asked to move when you have at least one legal move.
    The reversi class helps you think:
        game.legal_moves(turn)                  -> list of (x, y) you may play
        game.step(x, y, turn, commit = False)   -> how many pieces that move flips, board untouched
        game.step(x, y, turn)                   -> actually play it (copy the board first when searching ahead)
    """
    return miniMax(turn, 3).chooseMove(board, turn)

    
class miniMax:
    def __init__(self, me, depth):
        self.me = me
        self.depth = depth

    def search(self, board, turn, depth, score):
        if depth == 0:
            return score(board, self.me)
        game = reversi()
        game.board = board.copy()
        moves = game.legal_moves(turn)

        if len(moves) == 0:
            if len(game.legal_moves(-turn)) == 0:
                return score(board, self.me)
            return self.search(board, -turn, depth-1, score)
        
        values = []
        for x, y in moves:
            child = reversi()
            child.board = board.copy()
            child.step(x, y, turn)
            values.append(self.search(child.board, -turn, depth-1, score))
        if turn == self.me:
            return max(values)
        return min(values)
    
    def bestMove(self, board, turn, score):
        game = reversi()
        game.board = board.copy()

        best_move = (-1, -1)
        best_value = float('-inf')
        for x, y in game.legal_moves(turn):
            child = reversi()
            child.board = board.copy()
            child.step(x, y, turn)
            value = self.search(child.board, -turn, self.depth-1, score)
            if value > best_value:
                best_value = value
                best_move = (x, y)
        return best_move
    
    def chooseMove(self, board, turn):
        score = evaluation().chooseScoringFunction(board)
        return self.bestMove(board, turn, score)

class evaluation:
    def chooseScoringFunction(self, board):
        #add conditionals based on current state of game / if we can guess what opponent is using
        return self.combinedScore

    def pieceScore(self, board, me):
        return np.count_nonzero(board == me) - np.count_nonzero(board == -me)

    weights = np.array([
        [100, -20,  10,   5,   5,  10, -20, 100],
        [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
        [ 10,  -2,   1,   1,   1,   1,  -2,  10],
        [  5,  -2,   1,   0,   0,   1,  -2,   5],
        [  5,  -2,   1,   0,   0,   1,  -2,   5],
        [ 10,  -2,   1,   1,   1,   1,  -2,  10],
        [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
        [100, -20,  10,   5,   5,  10, -20, 100],
    ])
    #probably can just remove below and adjust weights ; currently if you remove corner score and only keep position score it does much worse
    def cornerScore(self, board, me):
        corners = [board[0, 0], board[0, 7], board[7, 0], board[7, 7]]
        return corners.count(me) - corners.count(-me)

    def mobilityScore(self, board, me):
        game = reversi()
        game.board = board.copy()
        return len(game.legal_moves(me)) - len(game.legal_moves(-me))

    def positionScore(self, board, me):
        return int(np.sum(self.weights * board)) * me

    def combinedScore(self, board, me):
        return (25 * self.cornerScore(board, me) + 5 * self.mobilityScore(board, me) + .1 * self.positionScore(board, me) + self.pieceScore(board, me))




