"""Square-based physical chess move detection and game logging.

The camera detector never decides whether a move is legal.  It identifies the
source and destination squares from the image change and the tracked physical
piece layout.  Every detected move is written to the raw logs.  A separate,
best-effort python-chess board is used only to produce standard SAN/PGN output.

Detection strategy
-------------------
Each square is inspected for an actual piece-shaped *object* rather than a raw
pixel-difference percentage.  For a single frame, ``analyze_square_occupancy``
crops each calibrated polygon, Otsu-thresholds it, and treats whichever side
of the threshold is the minority of the square as the candidate piece (since
the board material, not the piece, should dominate the polygon -- this works
for a piece lighter or darker than the square underneath it).  It then keeps
that candidate only if it is large enough and round/solid enough to plausibly
be a piece, rejecting thin slivers of shadow, grid lines, board texture, or a
near-uniform square with nothing on it at all.

A move is then found by comparing the *occupancy* of every square between the
"before" and "after" photos:

- A square that had an object and now has none is a candidate source.
- A square that had no object and now has one is a candidate destination
  (a normal move to an empty square).
- A square that had an object before *and* after, but changed a lot in
  between, is a candidate destination for a capture (the piece there was
  swapped out for a different one).

This is intentionally stricter than counting changed pixels: a shadow or a
lighting flicker changes pixels but does not make a new blob appear where
there wasn't one, so it is far less likely to be mistaken for a move.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


GridKey = Tuple[int, int]
Polygon = Sequence[Tuple[int, int]]


def grid_to_square(grid_key: GridKey) -> str:
    """Map the measured grid orientation to algebraic coordinates.

    The physical calibration uses:
        (0, 0) -> a8
        (0, 7) -> a1
        (7, 7) -> h1
    """

    file_index, rank_index = grid_key
    if not (0 <= file_index < 8 and 0 <= rank_index < 8):
        raise ValueError(f"Grid key outside the 8x8 board: {grid_key}")
    return f"{chr(ord('a') + file_index)}{8 - rank_index}"


def square_to_grid(square: str) -> GridKey:
    """Inverse of :func:`grid_to_square`."""

    if not re.fullmatch(r"[a-h][1-8]", square):
        raise ValueError(f"Invalid chess square: {square!r}")
    return ord(square[0]) - ord("a"), 8 - int(square[1])


def build_grid_map(board_squares: Mapping[GridKey, Polygon]) -> Dict[GridKey, str]:
    """Validate the hard-coded grid and return all 64 algebraic labels."""

    expected = {(file_index, rank_index) for file_index in range(8) for rank_index in range(8)}
    actual = set(board_squares)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"BOARD_SQUARES must contain exactly 64 squares; missing={missing}, extra={extra}")
    return {grid_key: grid_to_square(grid_key) for grid_key in board_squares}


###################################################################################
## Object/piece occupancy detection (single frame)
###################################################################################


@dataclass(frozen=True)
class SquareObservation:
    """Result of looking for a piece-shaped object inside one square, in one frame."""

    occupied: bool
    fill_ratio: float          # detected blob area / square area
    circularity: float         # 1.0 = perfect circle/ellipse, ~0 = thin sliver or line
    contour_center: Optional[Tuple[float, float]]


def _crop_polygon(image: np.ndarray, polygon: Polygon) -> Tuple[np.ndarray, np.ndarray]:
    """Crop the polygon's bounding box and return (crop, binary mask-in-crop)."""

    import cv2

    pts = np.asarray(polygon, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    x0, y0 = max(x, 0), max(y, 0)
    x1 = min(x + w, image.shape[1])
    y1 = min(y + h, image.shape[0])
    crop = image[y0:y1, x0:x1]
    mask = np.zeros((max(y1 - y0, 1), max(x1 - x0, 1)), dtype=np.uint8)
    shifted = (pts - [x0, y0]).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [shifted], 255)
    return crop, mask


def analyze_square_occupancy(
    frame: np.ndarray,
    polygon: Polygon,
    min_area_ratio: float = 0.10,
    min_circularity: float = 0.30,
    border_margin: int = 3,
    min_texture_std: float = 6.0,
    max_area_ratio: float = 0.85,
) -> SquareObservation:
    """Look for a piece-shaped object inside one calibrated square polygon.

    Unlike a plain frame-to-frame pixel diff, this inspects a *single* frame
    and searches for a blob that is plausibly a piece: large enough relative
    to the square, round/solid enough (not a thin line from a grid edge or a
    shadow gradient), and a minority of the square's pixels (since the board
    material, not the piece, should dominate the polygon). This works for
    both light pieces on dark squares and dark pieces on light squares.
    """

    import cv2

    crop, mask = _crop_polygon(frame, polygon)
    if crop.size == 0 or mask.size == 0:
        return SquareObservation(False, 0.0, 0.0, None)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if border_margin > 0:
        kernel_size = 2 * border_margin + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

    square_area = float(np.count_nonzero(mask))
    if square_area == 0:
        return SquareObservation(False, 0.0, 0.0, None)

    # A square with (near-)uniform brightness has no object on it, only flat
    # board material. Skip straight to "empty" -- this also protects against
    # Otsu's threshold degenerating on a constant image and marking the whole
    # square as one big "foreground" blob.
    if float(np.std(gray[mask > 0])) < min_texture_std:
        return SquareObservation(False, 0.0, 0.0, None)

    # Otsu picks a brightness threshold that best separates the square into
    # two groups, but it says nothing about *which* group is the piece. A
    # piece normally covers a minority of the square (the board material
    # dominates), so treat whichever side of the threshold covers fewer
    # pixels as the candidate object -- this works whether the piece is
    # lighter or darker than the square underneath it, without the earlier
    # bug of accidentally treating the whole bright background as "the
    # object" just because it happened to land above the threshold.
    _, above_threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    above_threshold = cv2.bitwise_and(above_threshold, above_threshold, mask=mask)
    above_count = int(np.count_nonzero(above_threshold))
    below_count = int(square_area) - above_count

    if above_count <= below_count:
        candidate_mask = above_threshold
    else:
        candidate_mask = cv2.bitwise_and(mask, cv2.bitwise_not(above_threshold))

    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_area_ratio = 0.0
    best_circularity = 0.0
    best_center: Optional[Tuple[float, float]] = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 4:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        area_ratio = area / square_area
        if area_ratio > max_area_ratio:
            # A blob nearly as large as the whole square is more likely a
            # thresholding artifact than a real piece silhouette.
            continue
        circularity = min(1.0, 4 * np.pi * area / (perimeter ** 2))
        if area_ratio > best_area_ratio:
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            best_area_ratio = area_ratio
            best_circularity = circularity
            best_center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])

    occupied = best_area_ratio >= min_area_ratio and best_circularity >= min_circularity
    return SquareObservation(occupied, best_area_ratio, best_circularity, best_center)


###################################################################################
## Frame-to-frame pixel change (kept only to confirm a capture on an already
## occupied square, where occupancy alone can't tell that the piece changed)
###################################################################################


@dataclass(frozen=True)
class SquareChange:
    grid_key: GridKey
    square: str
    changed_ratio: float
    mean_delta: float


def score_square_changes(
    before: np.ndarray,
    after: np.ndarray,
    board_squares: Mapping[GridKey, Polygon],
    pixel_threshold: int = 18,
    border_margin: int = 3,
) -> Dict[GridKey, SquareChange]:
    """Return a raw pixel-change ratio per square. Used only as a secondary
    signal to confirm captures on squares that stay occupied before and after."""

    import cv2

    if before.shape != after.shape:
        raise ValueError(f"Frame sizes do not match: {before.shape} vs {after.shape}")

    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY) if before.ndim == 3 else before
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY) if after.ndim == 3 else after
    before_gray = cv2.GaussianBlur(before_gray, (5, 5), 0)
    after_gray = cv2.GaussianBlur(after_gray, (5, 5), 0)
    difference = cv2.absdiff(before_gray, after_gray)

    results: Dict[GridKey, SquareChange] = {}
    for grid_key, corners in board_squares.items():
        mask = np.zeros(difference.shape, dtype=np.uint8)
        polygon = np.asarray(corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [polygon], 255)
        if border_margin > 0:
            kernel_size = 2 * border_margin + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

        values = difference[mask > 0]
        if values.size == 0:
            changed_ratio = 0.0
            mean_delta = 0.0
        else:
            changed = values >= pixel_threshold
            changed_ratio = float(np.mean(changed))
            mean_delta = float(np.mean(values[changed])) if np.any(changed) else 0.0

        results[grid_key] = SquareChange(
            grid_key=grid_key,
            square=grid_to_square(grid_key),
            changed_ratio=changed_ratio,
            mean_delta=mean_delta,
        )

    return results


###################################################################################
## Occupancy transitions between two frames -- this is the actual move signal
###################################################################################


@dataclass(frozen=True)
class SquareTransition:
    grid_key: GridKey
    square: str
    transition: str  # "vacated" | "filled" | "swapped" | "unchanged"
    before_occupied: bool
    after_occupied: bool
    fill_ratio_before: float
    fill_ratio_after: float
    changed_ratio: float   # secondary pixel-diff signal, used for captures + status display
    mean_delta: float
    score: float            # ranking score used to order candidates


def score_square_transitions(
    before: np.ndarray,
    after: np.ndarray,
    board_squares: Mapping[GridKey, Polygon],
    min_area_ratio: float = 0.10,
    min_circularity: float = 0.30,
    pixel_threshold: int = 18,
    capture_diff_ratio: float = 0.05,
) -> Tuple[SquareTransition, ...]:
    """Classify every square as vacated / filled / swapped / unchanged.

    "vacated" and "filled" come directly from object detection (a piece blob
    disappeared or appeared). "swapped" covers captures: the square held an
    object both before and after, but the pixel content underneath changed
    enough that the piece there is very unlikely to be the same one.
    """

    diff_scores = score_square_changes(before, after, board_squares, pixel_threshold=pixel_threshold)

    transitions = []
    for grid_key, corners in board_squares.items():
        before_obs = analyze_square_occupancy(before, corners, min_area_ratio, min_circularity)
        after_obs = analyze_square_occupancy(after, corners, min_area_ratio, min_circularity)
        diff = diff_scores[grid_key]

        if before_obs.occupied and not after_obs.occupied:
            transition = "vacated"
        elif not before_obs.occupied and after_obs.occupied:
            transition = "filled"
        elif before_obs.occupied and after_obs.occupied and diff.changed_ratio >= capture_diff_ratio:
            transition = "swapped"
        else:
            transition = "unchanged"

        # Ranking score: prioritize squares with a clear occupancy flip, and
        # use fill amount + pixel change to break ties among several flips.
        base = 2.0 if transition in ("vacated", "filled") else (1.0 if transition == "swapped" else 0.0)
        score = base + diff.changed_ratio + after_obs.fill_ratio * 0.5 + before_obs.fill_ratio * 0.5

        transitions.append(
            SquareTransition(
                grid_key=grid_key,
                square=grid_to_square(grid_key),
                transition=transition,
                before_occupied=before_obs.occupied,
                after_occupied=after_obs.occupied,
                fill_ratio_before=before_obs.fill_ratio,
                fill_ratio_after=after_obs.fill_ratio,
                changed_ratio=diff.changed_ratio,
                mean_delta=diff.mean_delta,
                score=score,
            )
        )

    return tuple(sorted(transitions, key=lambda t: t.score, reverse=True))


@dataclass(frozen=True)
class DetectedMove:
    source: str
    destination: str
    uci: str
    piece: str
    captured_piece: str | None
    changed_squares: Tuple[SquareTransition, ...]


class MoveDetectionError(RuntimeError):
    """Raised when the camera change cannot be reduced to one move."""

    def __init__(self, message: str, changes: Iterable[SquareTransition] = ()) -> None:
        super().__init__(message)
        self.changes = tuple(changes)


class PhysicalPosition:
    """Piece placement used for source/destination inference, without validation."""

    def __init__(self) -> None:
        self.pieces: MutableMapping[str, str] = {}
        self.turn = "white"
        self.reset()

    def reset(self) -> None:
        self.pieces.clear()
        for file_name, piece in zip("abcdefgh", "rnbqkbnr"):
            self.pieces[file_name + "8"] = piece
            self.pieces[file_name + "7"] = "p"
            self.pieces[file_name + "2"] = "P"
            self.pieces[file_name + "1"] = piece.upper()
        self.turn = "white"

    def piece_at(self, square: str) -> str | None:
        return self.pieces.get(square)

    def belongs_to_turn(self, piece: str | None) -> bool:
        if piece is None:
            return False
        return piece.isupper() if self.turn == "white" else piece.islower()

    def apply(self, source: str, destination: str) -> Tuple[str, str | None]:
        """Apply a detected move without checking whether chess rules allow it."""

        piece = self.pieces.get(source)
        if piece is None:
            raise ValueError(f"No tracked piece is present on {source}")

        captured_piece = self.pieces.get(destination)

        # Physical castling moves both king and rook, although the log stores
        # the king move as required by UCI and PGN.
        if piece.lower() == "k" and source in ("e1", "e8"):
            rook_moves = {
                ("e1", "g1"): ("h1", "f1"),
                ("e1", "c1"): ("a1", "d1"),
                ("e8", "g8"): ("h8", "f8"),
                ("e8", "c8"): ("a8", "d8"),
            }
            rook_move = rook_moves.get((source, destination))
            if rook_move:
                rook_source, rook_destination = rook_move
                rook = self.pieces.pop(rook_source, None)
                if rook is not None:
                    self.pieces[rook_destination] = rook

        # If a pawn moved diagonally to an empty square, remove the en-passant
        # capture square from the tracked physical layout.
        if piece.lower() == "p" and source[0] != destination[0] and captured_piece is None:
            captured_square = destination[0] + source[1]
            captured_piece = self.pieces.pop(captured_square, None)

        self.pieces.pop(source)

        # The camera cannot identify the selected promotion piece yet. Queen is
        # the explicit default so the raw coordinate remains exportable.
        if piece == "P" and destination[1] == "8":
            piece = "Q"
        elif piece == "p" and destination[1] == "1":
            piece = "q"

        self.pieces[destination] = piece
        self.turn = "black" if self.turn == "white" else "white"
        return piece, captured_piece


def detect_move(
    before: np.ndarray,
    after: np.ndarray,
    board_squares: Mapping[GridKey, Polygon],
    position: PhysicalPosition,
    min_area_ratio: float = 0.10,
    min_circularity: float = 0.30,
    pixel_threshold: int = 18,
    capture_diff_ratio: float = 0.05,
    max_candidates: int = 6,
) -> DetectedMove:
    """Detect one physical move from piece-object occupancy transitions.

    A square only becomes a source/destination candidate if a piece-shaped
    object actually appeared or disappeared there (or, for captures, the
    object present clearly changed) -- not merely because some pixels crossed
    a brightness threshold.
    """

    transitions = score_square_transitions(
        before,
        after,
        board_squares,
        min_area_ratio=min_area_ratio,
        min_circularity=min_circularity,
        pixel_threshold=pixel_threshold,
        capture_diff_ratio=capture_diff_ratio,
    )
    active = [t for t in transitions if t.transition != "unchanged"][:max_candidates]

    if len(active) < 2:
        raise MoveDetectionError(
            "Fewer than two squares show a piece object appearing, disappearing, or being swapped",
            transitions[:6],
        )

    vacated = [t for t in active if t.transition == "vacated"]
    filled_or_swapped = [t for t in active if t.transition in ("filled", "swapped")]

    sources = [t for t in vacated if position.belongs_to_turn(position.piece_at(t.square))]
    if not sources:
        raise MoveDetectionError(
            f"No square that lost a piece object contains the tracked {position.turn} piece",
            active,
        )

    # Castling changes four squares. If the king vacated and one of its two
    # possible destinations was filled, prefer that pair regardless of score.
    castling_pairs = {
        "e1": ("g1", "c1"),
        "e8": ("g8", "c8"),
    }
    filled_names = {t.square for t in filled_or_swapped}
    for source_t in sources:
        piece = position.piece_at(source_t.square)
        if piece and piece.lower() == "k" and source_t.square in castling_pairs:
            for destination in castling_pairs[source_t.square]:
                if destination in filled_names:
                    return DetectedMove(
                        source=source_t.square,
                        destination=destination,
                        uci=source_t.square + destination,
                        piece=piece,
                        captured_piece=position.piece_at(destination),
                        changed_squares=tuple(active),
                    )

    source_t = sources[0]
    source = source_t.square
    piece = position.piece_at(source)
    destination_candidates = [t for t in filled_or_swapped if t.square != source]

    # A destination should be empty or contain the other side. This rejects a
    # square that merely looks different because it still holds the mover's
    # own piece (e.g. a partial occlusion during the move).
    usable_destinations = []
    for t in destination_candidates:
        occupant = position.piece_at(t.square)
        if occupant is None or not position.belongs_to_turn(occupant):
            usable_destinations.append(t)
    if usable_destinations:
        destination_candidates = usable_destinations

    if not destination_candidates:
        raise MoveDetectionError("No destination square gained or swapped a piece object", active)

    destination_t = destination_candidates[0]
    destination = destination_t.square
    captured_piece = position.piece_at(destination)
    promotion = ""
    if piece and piece.lower() == "p" and destination[1] in ("1", "8"):
        promotion = "q"

    return DetectedMove(
        source=source,
        destination=destination,
        uci=source + destination + promotion,
        piece=piece or "?",
        captured_piece=captured_piece,
        changed_squares=tuple(active),
    )


class GameRecorder:
    """Write raw move records immediately and maintain a Chessigma-ready PGN."""

    def __init__(self, output_root: Path | str, session_name: str) -> None:
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", session_name.strip()).strip("_")
        if not safe_name:
            safe_name = datetime.now().strftime("game_%Y%m%d_%H%M%S")

        self.output_dir = Path(output_root) / safe_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.uci_moves = []
        self.san_moves = []
        self.pgn_synced = True
        self._pgn_board = None
        self._pgn_game = None
        self._pgn_node = None
        self._start_pgn()
        self.write_exports()

    def _start_pgn(self) -> None:
        try:
            import chess
            import chess.pgn
        except ImportError:
            self.pgn_synced = False
            return

        self._chess = chess
        self._pgn_board = chess.Board()
        self._pgn_game = chess.pgn.Game()
        self._pgn_game.headers["Event"] = "Camera-recorded game"
        self._pgn_game.headers["Site"] = "Physical chessboard"
        self._pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        self._pgn_game.headers["Round"] = "-"
        self._pgn_game.headers["White"] = "White"
        self._pgn_game.headers["Black"] = "Black"
        self._pgn_game.headers["Result"] = "*"
        self._pgn_node = self._pgn_game

    def record(self, move: DetectedMove, side: str) -> str:
        """Record a move unconditionally; return SAN or the raw UCI fallback."""

        san = ""
        pgn_status = "not available"
        if self.pgn_synced and self._pgn_board is not None:
            try:
                chess_move = self._chess.Move.from_uci(move.uci)
                san = self._pgn_board.san(chess_move)
                self._pgn_node = self._pgn_node.add_variation(chess_move)
                self._pgn_board.push(chess_move)
                self.san_moves.append(san)
                pgn_status = "converted"
            except (AssertionError, ValueError):
                # This affects only PGN conversion. The physical move remains
                # in every raw export and later moves continue to be recorded.
                self.pgn_synced = False
                pgn_status = "raw move saved; PGN conversion paused"

        self.uci_moves.append(move.uci)
        row = {
            "ply": len(self.uci_moves),
            "move_number": (len(self.uci_moves) + 1) // 2,
            "side": side,
            "source": move.source,
            "destination": move.destination,
            "uci": move.uci,
            "san": san,
            "piece": move.piece,
            "captured_piece": move.captured_piece or "",
            "pgn_status": pgn_status,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "changed_squares": " ".join(
                f"{change.square}:{change.transition}:{change.changed_ratio:.3f}"
                for change in move.changed_squares
            ),
        }
        self.rows.append(row)
        self.write_exports()
        return san or move.uci

    def write_exports(self) -> None:
        """Rewrite all small exports after every move for crash-safe review."""

        (self.output_dir / "moves_uci.txt").write_text(
            " ".join(self.uci_moves) + ("\n" if self.uci_moves else ""),
            encoding="utf-8",
        )

        fieldnames = [
            "ply",
            "move_number",
            "side",
            "source",
            "destination",
            "uci",
            "san",
            "piece",
            "captured_piece",
            "pgn_status",
            "timestamp",
            "changed_squares",
        ]
        with (self.output_dir / "moves.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

        (self.output_dir / "moves.json").write_text(
            json.dumps(self.rows, indent=2),
            encoding="utf-8",
        )

        if self._pgn_game is not None:
            pgn_text = str(self._pgn_game).rstrip() + "\n"
        else:
            pgn_text = (
                "[Event \"Camera-recorded game\"]\n"
                "[Result \"*\"]\n\n"
                "{Install python-chess to generate standard PGN notation.} *\n"
            )
        (self.output_dir / "game.pgn").write_text(pgn_text, encoding="utf-8")
