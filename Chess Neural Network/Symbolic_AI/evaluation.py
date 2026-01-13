import chess

def evaluate_board(board) -> int:
    pieces = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }

    if board.is_checkmate():
        return float('-inf') if board.turn == chess.WHITE else float('inf')

    if board.is_stalemate() or board.is_insufficient_material() \
        or board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
        return 0

    evaluation = 0
    for piece_type, value in pieces.items():
        evaluation += len(board.pieces(piece_type, chess.WHITE)) * value
        evaluation -= len(board.pieces(piece_type, chess.BLACK)) * value

    return evaluation