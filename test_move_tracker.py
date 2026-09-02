import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


MODULE_PATH = Path(__file__).with_name("move_tracker.py")
SPEC = importlib.util.spec_from_file_location("move_tracker", MODULE_PATH)
move_tracker = importlib.util.module_from_spec(SPEC)
import sys
sys.modules["move_tracker"] = move_tracker
SPEC.loader.exec_module(move_tracker)


SQUARE_SIZE = 60


def make_square_polygon(col, row, size=SQUARE_SIZE):
    x, y = col * size, row * size
    return [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]


def make_board_image(board_color=200, cols=3, rows=3, size=SQUARE_SIZE):
    """A flat, empty board: just a solid background, no pieces anywhere."""
    return np.full((rows * size, cols * size, 3), board_color, dtype=np.uint8)


def draw_piece(image, col, row, piece_color=40, radius_ratio=0.35, size=SQUARE_SIZE):
    """Draw a circular 'piece' silhouette roughly centered on one square."""
    cx = col * size + size // 2
    cy = row * size + size // 2
    radius = int(size * radius_ratio)
    cv2.circle(image, (cx, cy), radius, (piece_color, piece_color, piece_color), -1)
    return image


def transition(square, kind, changed_ratio=0.0):
    """Build a SquareTransition for tests that exercise detect_move's
    candidate-selection logic without re-deriving it from pixels."""
    grid_key = move_tracker.square_to_grid(square)
    return move_tracker.SquareTransition(
        grid_key=grid_key,
        square=square,
        transition=kind,
        before_occupied=(kind in ("vacated", "swapped")),
        after_occupied=(kind in ("filled", "swapped")),
        fill_ratio_before=0.3 if kind in ("vacated", "swapped") else 0.0,
        fill_ratio_after=0.3 if kind in ("filled", "swapped") else 0.0,
        changed_ratio=changed_ratio,
        mean_delta=80.0,
        score=(2.0 if kind in ("vacated", "filled") else 1.0) + changed_ratio,
    )


class OccupancyDetectionTests(unittest.TestCase):
    """These run the real image-processing code on synthetic pixel data --
    no mocking -- to confirm the object detector actually finds a piece-shaped
    blob and actually rejects an empty square."""

    def test_empty_square_is_not_occupied(self):
        board = make_board_image()
        polygon = make_square_polygon(1, 1)
        observation = move_tracker.analyze_square_occupancy(board, polygon)
        self.assertFalse(observation.occupied)

    def test_square_with_a_round_piece_is_occupied(self):
        board = make_board_image()
        draw_piece(board, col=1, row=1)
        polygon = make_square_polygon(1, 1)
        observation = move_tracker.analyze_square_occupancy(board, polygon)
        self.assertTrue(observation.occupied)
        self.assertGreater(observation.fill_ratio, 0.10)
        self.assertGreater(observation.circularity, 0.30)

    def test_thin_shadow_line_is_not_mistaken_for_a_piece(self):
        board = make_board_image()
        # A thin diagonal line simulates a shadow/grid-line artifact: it can
        # cover plenty of pixels but has very low circularity.
        cv2.line(board, (60, 62), (118, 118), (40, 40, 40), thickness=2)
        polygon = make_square_polygon(1, 1)
        observation = move_tracker.analyze_square_occupancy(board, polygon)
        self.assertFalse(observation.occupied)

    def test_vacated_and_filled_transition_from_real_images(self):
        before = make_board_image()
        draw_piece(before, col=0, row=0)  # piece starts on square (0,0)

        after = make_board_image()
        draw_piece(after, col=2, row=2)  # same piece now on square (2,2)

        board_squares = {
            (0, 0): make_square_polygon(0, 0),
            (2, 2): make_square_polygon(2, 2),
            (1, 1): make_square_polygon(1, 1),
        }
        transitions = {
            t.grid_key: t.transition
            for t in move_tracker.score_square_transitions(before, after, board_squares)
        }
        self.assertEqual(transitions[(0, 0)], "vacated")
        self.assertEqual(transitions[(2, 2)], "filled")
        self.assertEqual(transitions[(1, 1)], "unchanged")


class MoveTrackerTests(unittest.TestCase):
    def test_grid_orientation(self):
        self.assertEqual(move_tracker.grid_to_square((0, 0)), "a8")
        self.assertEqual(move_tracker.grid_to_square((0, 7)), "a1")
        self.assertEqual(move_tracker.grid_to_square((7, 7)), "h1")
        self.assertEqual(move_tracker.square_to_grid("e2"), (4, 6))

    def test_normal_move_detection(self):
        position = move_tracker.PhysicalPosition()
        ranked = (transition("e2", "vacated"), transition("e4", "filled"))
        with patch.object(move_tracker, "score_square_transitions", return_value=ranked):
            detected = move_tracker.detect_move(
                np.zeros((1, 1, 3), dtype=np.uint8),
                np.zeros((1, 1, 3), dtype=np.uint8),
                {},
                position,
            )
        self.assertEqual(detected.uci, "e2e4")
        position.apply(detected.source, detected.destination)
        self.assertEqual(position.piece_at("e4"), "P")
        self.assertEqual(position.turn, "black")

    def test_capture_uses_current_side_as_source(self):
        position = move_tracker.PhysicalPosition()
        position.apply("e2", "e4")
        position.apply("d7", "d5")
        # e4 takes d5: e4 vacates, d5 keeps an object but it swapped (capture).
        ranked = (transition("d5", "swapped", changed_ratio=0.4), transition("e4", "vacated"))
        with patch.object(move_tracker, "score_square_transitions", return_value=ranked):
            detected = move_tracker.detect_move(
                np.zeros((1, 1, 3), dtype=np.uint8),
                np.zeros((1, 1, 3), dtype=np.uint8),
                {},
                position,
            )
        self.assertEqual(detected.uci, "e4d5")
        self.assertEqual(detected.captured_piece, "p")

    def test_castling_prefers_king_move_among_four_changed_squares(self):
        position = move_tracker.PhysicalPosition()
        position.pieces.pop("f1")
        position.pieces.pop("g1")
        ranked = (
            transition("h1", "vacated"),
            transition("f1", "filled"),
            transition("e1", "vacated"),
            transition("g1", "filled"),
        )
        with patch.object(move_tracker, "score_square_transitions", return_value=ranked):
            detected = move_tracker.detect_move(
                np.zeros((1, 1, 3), dtype=np.uint8),
                np.zeros((1, 1, 3), dtype=np.uint8),
                {},
                position,
            )
        self.assertEqual(detected.uci, "e1g1")
        position.apply(detected.source, detected.destination)
        self.assertEqual(position.piece_at("g1"), "K")
        self.assertEqual(position.piece_at("f1"), "R")

    def test_no_move_detected_when_nothing_changes(self):
        position = move_tracker.PhysicalPosition()
        ranked = (transition("e2", "unchanged"), transition("e4", "unchanged"))
        with patch.object(move_tracker, "score_square_transitions", return_value=ranked):
            with self.assertRaises(move_tracker.MoveDetectionError):
                move_tracker.detect_move(
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    {},
                    position,
                )

    def test_raw_exports_are_written_without_pgn_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(move_tracker.GameRecorder, "_start_pgn", lambda self: None):
                recorder = move_tracker.GameRecorder(directory, "test game")
                recorder.pgn_synced = False
                detected = move_tracker.DetectedMove(
                    source="e2",
                    destination="e4",
                    uci="e2e4",
                    piece="P",
                    captured_piece=None,
                    changed_squares=(transition("e2", "vacated"), transition("e4", "filled")),
                )
                recorder.record(detected, "white")
                output = Path(directory) / "test_game"
                self.assertEqual((output / "moves_uci.txt").read_text(), "e2e4\n")
                self.assertIn("e2e4", (output / "moves.csv").read_text())
                self.assertIn('"uci": "e2e4"', (output / "moves.json").read_text())


if __name__ == "__main__":
    unittest.main()
