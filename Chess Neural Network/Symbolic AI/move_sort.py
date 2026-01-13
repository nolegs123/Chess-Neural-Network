import chess

def order_moves(board):
    piece_value = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 100
    }

    captures = []
    checks = []
    quiet = []

    for move in board.legal_moves:
        if board.is_capture(move):
            captures.append(move)
        else:
            board.push(move)
            if board.is_check():
                checks.append(move)
            else:
                quiet.append(move)
            board.pop()

    captures.sort(
        key=lambda m: piece_value.get(
            board.piece_at(m.to_square).piece_type
            if board.piece_at(m.to_square) else 0,
            0
        ),
        reverse=True
    )

    return captures + checks + quiet