# SPIS Final — Physical Chess Move Recorder

This Raspberry Pi/OpenCV project watches a fixed physical chessboard, maps its
64 hand-calibrated polygons to chess coordinates, and records each move without
using legality as a gate.

Move detection looks for an actual piece-shaped **object** in each square
(a blob that's the right size and round/solid enough to be a piece), not just
"how many pixels changed." A square becomes a source when its piece object
disappears, a destination when one appears, and a capture when the object on
an already-occupied square is swapped for a different one. This is more
resistant to shadows, lighting flicker, and camera noise than a plain
before/after pixel diff, since those change pixel brightness but don't make a
piece-shaped blob appear or disappear where there wasn't one.

Grid orientation:

- `(0, 0) = a8`
- `(0, 7) = a1`
- `(7, 7) = h1`
- General rule: `(file_index, rank_index) -> abcdefgh[file_index] + (8-rank_index)`

Run `open chess/main.py`. After alignment, make one move, remove your hands
from the camera view, and press Space. The program writes these files after
every detected move:

- `game.pgn` — standard PGN for upload or paste into Chessigma
- `moves_uci.txt` — unconditional raw coordinate moves such as `e2e4`
- `moves.csv` — detailed move and image-change log
- `moves.json` — the same detailed log in JSON form

The raw logs are authoritative: a move is still saved if standard PGN notation
cannot be generated. See `how to run?` for setup and controls.
