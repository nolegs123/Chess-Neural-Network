"""Dataset utilities for loading FEN sequences into tensors.

- Reads all FEN_moves_*.txt files under FENs/.
- Optionally derives the next-move label by comparing consecutive FENs.
- Encodes boards into channel-first tensors suitable for CNNs.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import chess
import torch
from torch.utils.data import Dataset

# Plane layout: 12 piece planes (white then black), 1 side-to-move plane,
# 4 castling planes (WK, WQ, BK, BQ), 1 en-passant file plane. Total: 18.
PLANE_COUNT = 18

# Map (color, piece_type) -> plane index.
PIECE_PLANES = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}

@dataclass
class Sample:
    fen: str
    move_uci: Optional[str]  # None when labels are not requested


def load_fen_lines(path: Path) -> List[str]:
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


def infer_next_move(fen: str, next_fen: str) -> Optional[chess.Move]:
    """Find the legal move that transforms fen -> next_fen, else None."""
    board = chess.Board(fen)
    for move in board.legal_moves:
        test_board = board.copy()
        test_board.push(move)
        if test_board.fen() == next_fen:
            return move
    return None


def fen_to_tensor(fen: str) -> torch.Tensor:
    """Encode a FEN into an (18, 8, 8) float tensor."""
    board = chess.Board(fen)
    planes = torch.zeros((PLANE_COUNT, 8, 8), dtype=torch.float32)

    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)  # flip to have white at bottom
        col = chess.square_file(square)
        planes[PIECE_PLANES[(piece.color, piece.piece_type)], row, col] = 1.0

    # Side to move plane: all ones if white to move, zeros otherwise.
    if board.turn == chess.WHITE:
        planes[12].fill_(1.0)

    # Castling planes (one-hot across board for simplicity).
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13].fill_(1.0)
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14].fill_(1.0)
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15].fill_(1.0)
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16].fill_(1.0)

    # En-passant file plane: mark the target file if available.
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        planes[17, :, ep_file] = 1.0

    return planes


class FENDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        fen_glob: str = "FENs/FEN_moves_*.txt",
        with_labels: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        max_positions: Optional[int] = None,
    ) -> None:
        self.root = Path(root_dir)
        self.transform = transform
        self.samples: List[Sample] = []

        paths = sorted(self.root.glob(fen_glob))
        for path in paths:
            fens = load_fen_lines(path)
            if with_labels:
                for cur, nxt in zip(fens, fens[1:]):
                    move = infer_next_move(cur, nxt)
                    if move is None:
                        continue  # skip if alignment fails
                    self.samples.append(Sample(cur, move.uci()))
            else:
                for cur in fens:
                    self.samples.append(Sample(cur, None))

        if max_positions is not None:
            self.samples = self.samples[:max_positions]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        x = fen_to_tensor(sample.fen)
        if self.transform:
            x = self.transform(x)
        return {
            "x": x,               # Tensor (18, 8, 8)
            "fen": sample.fen,    # Raw FEN string
            "move": sample.move_uci,  # UCI move string or None
        }


def collate_fn(batch: Sequence[dict]):
    xs = torch.stack([item["x"] for item in batch])
    moves = [item["move"] for item in batch]
    fens = [item["fen"] for item in batch]
    return {"x": xs, "move": moves, "fen": fens}


def make_dataloader(
    root_dir: str | Path,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    with_labels: bool = True,
    max_positions: Optional[int] = None,
):
    ds = FENDataset(root_dir, with_labels=with_labels, max_positions=max_positions)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)


if __name__ == "__main__":
    # Quick smoke test
    loader = make_dataloader(Path(__file__).parent, batch_size=2, max_positions=4)
    batch = next(iter(loader))
    print(batch["x"].shape, batch["move"])
