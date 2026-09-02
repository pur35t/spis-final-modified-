"""Live Raspberry Pi camera application for recording a physical chess game."""

###################################################################################
## Import Libraries
###################################################################################
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2

###################################################################################
## Import Local Modules
###################################################################################
from move_tracker import (
    GameRecorder,
    MoveDetectionError,
    PhysicalPosition,
    build_grid_map,
    detect_move,
    grid_to_square,
)

###################################################################################
## Camera Configuration & Predefined Grid
###################################################################################
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

picam2 = Picamera2()
camera_config = picam2.create_video_configuration(
    main={"format": "BGR888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
)
picam2.configure(camera_config)
picam2.start()

def capture_frame():
    return picam2.capture_array()

# Pre-defined 64 square coordinates: (top-left, top-right, bottom-right, bottom-left)
BOARD_SQUARES = {
     (0,0):  [(566, 1), (516, 1), (514, 58), (565, 59)],
     (0,1):  [(516, 1), (461, 1), (458, 57), (514, 58)],
     (0,2):  [(461, 1), (392, 1), (391, 55), (458, 57)],
     (0,3):  [(392, 1), (329, 1), (328, 54), (391, 55)],
     (0,4):  [(329, 1), (258, 1), (258, 53), (328, 54)],
     (0,5):  [(258, 1), (195, 1), (195, 52), (258, 53)],
     (0,6):  [(195, 1), (130, 1), (130, 51), (195, 52)],
     (0,7):  [(130, 1), (66, 1), (67, 50), (130, 51)],
     (1,0):  [(565, 59), (514, 58), (512, 123), (564, 123)],
     (1,1):  [(514, 58), (458, 57), (456, 122), (512, 123)],
     (1,2):  [(458, 57), (391, 55), (390, 121), (456, 122)],
     (1,3):  [(391, 55), (328, 54), (327, 120), (390, 121)],
     (1,4):  [(328, 54), (258, 53), (259, 119), (327, 120)],
     (1,5):  [(258, 53), (195, 52), (195, 118), (259, 119)],
     (1,6):  [(195, 52), (130, 51), (130, 117), (195, 118)],
     (1,7):  [(130, 51), (67, 50), (68, 116), (130, 117)],
     (2,0):  [(564, 123), (512, 123), (510, 182), (563, 182)],
     (2,1):  [(512, 123), (456, 122), (454, 181), (510, 182)],
     (2,2):  [(456, 122), (390, 121), (389, 180), (454, 181)],
     (2,3):  [(390, 121), (327, 120), (326, 179), (389, 180)],
     (2,4):  [(327, 120), (259, 119), (259, 178), (326, 179)],
     (2,5):  [(259, 119), (195, 118), (196, 177), (259, 178)],
     (2,6):  [(195, 118), (130, 117), (130, 176), (196, 177)],
     (2,7):  [(130, 117), (68, 116), (69, 176), (130, 176)],
     (3,0):  [(563, 182), (510, 182), (508, 245), (562, 245)],
     (3,1):  [(510, 182), (454, 181), (451, 245), (508, 245)],
     (3,2):  [(454, 181), (389, 180), (388, 245), (451, 245)],
     (3,3):  [(389, 180), (326, 179), (325, 245), (388, 245)],
     (3,4):  [(326, 179), (259, 178), (260, 244), (325, 245)],
     (3,5):  [(259, 178), (196, 177), (196, 244), (260, 244)],
     (3,6):  [(196, 177), (130, 176), (130, 244), (196, 244)],
     (3,7):  [(130, 176), (69, 176), (70, 244), (130, 244)],
     (4,0):  [(562, 245), (508, 245), (506, 311), (561, 311)],
     (4,1):  [(508, 245), (451, 245), (449, 311), (506, 311)],
     (4,2):  [(451, 245), (388, 245), (388, 310), (449, 311)],
     (4,3):  [(388, 245), (325, 245), (325, 310), (388, 310)],
     (4,4):  [(325, 245), (260, 244), (261, 310), (325, 310)],
     (4,5):  [(260, 244), (196, 244), (196, 309), (261, 310)],
     (4,6):  [(196, 244), (130, 244), (130, 309), (196, 309)],
     (4,7):  [(130, 244), (70, 244), (70, 309), (130, 309)],
     (5,0):  [(561, 311), (506, 311), (504, 377), (560, 377)],
     (5,1):  [(506, 311), (449, 311), (446, 376), (504, 377)],
     (5,2):  [(449, 311), (388, 310), (387, 375), (446, 376)],
     (5,3):  [(388, 310), (325, 310), (324, 375), (387, 375)],
     (5,4):  [(325, 310), (261, 310), (261, 374), (324, 375)],
     (5,5):  [(261, 310), (196, 309), (197, 373), (261, 374)],
     (5,6):  [(196, 309), (130, 309), (130, 372), (197, 373)],
     (5,7):  [(130, 309), (70, 309), (71, 372), (130, 372)],
     (6,0):  [(560, 377), (504, 377), (503, 434), (559, 434)],
     (6,1):  [(504, 377), (446, 376), (444, 434), (503, 434)],
     (6,2):  [(446, 376), (387, 375), (386, 434), (444, 434)],
     (6,3):  [(387, 375), (324, 375), (323, 434), (386, 434)],
     (6,4):  [(324, 375), (261, 374), (262, 434), (323, 434)],
     (6,5):  [(261, 374), (197, 373), (197, 434), (262, 434)],
     (6,6):  [(197, 373), (130, 372), (130, 434), (197, 434)],
     (6,7):  [(130, 372), (71, 372), (72, 434), (130, 434)],
     (7,0):  [(559, 434), (503, 434), (501, 479), (559, 479)],
     (7,1):  [(503, 434), (444, 434), (442, 479), (501, 479)],
     (7,2):  [(444, 434), (386, 434), (385, 479), (442, 479)],
     (7,3):  [(386, 434), (323, 434), (322, 479), (385, 479)],
     (7,4):  [(323, 434), (262, 434), (263, 479), (322, 479)],
     (7,5):  [(262, 434), (197, 434), (198, 479), (263, 479)],
     (7,6):  [(197, 434), (130, 434), (131, 479), (198, 479)],
     (7,7):  [(130, 434), (72, 434), (73, 479), (131, 479)]
}

###################################################################################
## Detection and logging configuration
###################################################################################
# The detector looks for an actual piece-shaped blob in each square (not just
# "these pixels changed"). A square only counts as occupied if a detected
# blob covers at least MIN_PIECE_AREA_RATIO of the square and is at least
# MIN_PIECE_CIRCULARITY round/solid (rejects thin shadow slivers and grid
# lines). Lower MIN_PIECE_AREA_RATIO if small/thin pieces (pawns, bishops)
# aren't being detected as present; raise it if shadows or board texture are
# being mistaken for a piece.
MIN_PIECE_AREA_RATIO = 0.10
MIN_PIECE_CIRCULARITY = 0.30

# Pixel-diff settings are now only used to catch captures: a square that has
# a piece both before and after a move, but a different one (e.g. a pawn
# taking a piece). CAPTURE_DIFF_RATIO is how much of the square must change
# for that to count as a swap rather than the same piece sitting still.
PIXEL_DIFF_THRESHOLD = 18
CAPTURE_DIFF_RATIO = 0.05

STABLE_FRAME_SAMPLES = 3

# This validates all 64 hard-coded polygons and fixes the physical orientation:
# (0, 0)=a8, (0, 7)=a1, and (7, 7)=h1.
GRID_TO_SQUARE = build_grid_map(BOARD_SQUARES)


def capture_stable_frame(sample_count=STABLE_FRAME_SAMPLES):
    """Average a few frames to reduce camera noise after a player moves."""

    frames = [capture_frame().astype(np.float32) for _ in range(sample_count)]
    return np.mean(frames, axis=0).astype(np.uint8)

def draw_predefined_grid(frame, square_dict, show_labels=True):
    """Draw every calibrated polygon and its verified algebraic label."""
    for grid_key, corners in square_dict.items():
        pts = np.array(corners, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
        if show_labels:
            center = np.mean(np.asarray(corners), axis=0).astype(int)
            cv2.putText(
                frame,
                grid_to_square(grid_key),
                (int(center[0]) - 10, int(center[1]) + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (0, 255, 255),
                1,
            )


def show_tracker(frame, position, recorder, last_move, status):
    """Show the calibrated board, recent move log, and simple controls."""

    display = frame.copy()
    draw_predefined_grid(display, BOARD_SQUARES)
    sidebar = np.zeros((FRAME_HEIGHT, 420, 3), dtype=np.uint8)
    display = np.concatenate((display, sidebar), axis=1)

    x = FRAME_WIDTH + 18
    cv2.putText(display, f"Turn: {position.turn.title()}", (x, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(display, "SPACE: record move", (x, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
    cv2.putText(display, "R: refresh reference", (x, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
    cv2.putText(display, "X or ESC: export and quit", (x, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
    cv2.putText(display, f"Last: {last_move or '-'}", (x, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 255, 0), 2)
    cv2.putText(display, "Recent moves", (x, 187), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 255), 1)

    recent_rows = recorder.rows[-10:]
    for row_number, row in enumerate(recent_rows):
        label = row["san"] or row["uci"]
        prefix = f'{row["move_number"]}.' if row["side"] == "white" else f'{row["move_number"]}...'
        cv2.putText(
            display,
            f"{prefix} {label}",
            (x, 214 + row_number * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
        )

    cv2.putText(display, status[:52], (10, FRAME_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 1)
    cv2.imshow("Chess Vision Tracker", display)

###################################################################################
## Preview Alignment Check
###################################################################################
print("Displaying grid overlay preview. Align board and press 'c' to continue...")
while True:
    frame = capture_frame()
    draw_predefined_grid(frame, BOARD_SQUARES)
    cv2.imshow("Align Physical Board with Grid", frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.destroyAllWindows()
        break

###################################################################################
## Main recording loop
###################################################################################
session_name = input("Enter Session/Run Name: ").strip()
output_root = Path(__file__).resolve().parent / "game_logs"
recorder = GameRecorder(output_root, session_name)
position = PhysicalPosition()
reference_frame = capture_stable_frame()
last_move = ""
status = "Ready. Move one piece, clear your hands, then press SPACE."

print(f"Verified grid: (0,0)={GRID_TO_SQUARE[(0,0)]}, (0,7)={GRID_TO_SQUARE[(0,7)]}, (7,7)={GRID_TO_SQUARE[(7,7)]}")
print(f"Saving move logs in: {recorder.output_dir}")

try:
    while True:
        live_frame = capture_frame()
        show_tracker(live_frame, position, recorder, last_move, status)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("x"), 27):
            break

        if key == ord("r"):
            reference_frame = capture_stable_frame()
            status = "Reference refreshed; no move was logged."
            print(status)
            continue

        if key != ord(" "):
            continue

        after_frame = capture_stable_frame()
        side = position.turn
        try:
            detected = detect_move(
                reference_frame,
                after_frame,
                BOARD_SQUARES,
                position,
                min_area_ratio=MIN_PIECE_AREA_RATIO,
                min_circularity=MIN_PIECE_CIRCULARITY,
                pixel_threshold=PIXEL_DIFF_THRESHOLD,
                capture_diff_ratio=CAPTURE_DIFF_RATIO,
            )
            position.apply(detected.source, detected.destination)
            display_move = recorder.record(detected, side)
            reference_frame = after_frame
            last_move = display_move
            changed = ", ".join(
                f"{change.square}={change.transition}"
                for change in detected.changed_squares[:4]
            )
            status = f"Logged {detected.uci}. Squares: {changed}"
            print(status)
        except MoveDetectionError as error:
            changed = ", ".join(
                f"{change.square}={change.transition}({change.changed_ratio:.1%})"
                for change in error.changes[:6]
            )
            status = f"Not logged: {error}. Top squares: {changed}"
            print(status)

finally:
    recorder.write_exports()
    picam2.stop()
    cv2.destroyAllWindows()
    print(f"Program exited. Exports are in: {recorder.output_dir}")
