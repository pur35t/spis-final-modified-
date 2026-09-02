import cv2
import numpy as np

def fen2board_black(fen_line):
    black_player_bool_position = []
    for row in fen_line.split(' ')[0].split('/'):
        bool_row = []
        for cell in list(row):
            if cell.isnumeric():
                for i in range(int(cell)):
                    bool_row.append(0)
            else:
                if cell.islower():
                    bool_row.append(1)
                else:
                    bool_row.append(0)
        black_player_bool_position.append(bool_row)
    return np.array(black_player_bool_position)

def point_in_quad(mid_point, quad_corners):
    """Returns True if mid_point (x, y) is inside the 4-corner polygon."""
    pts = np.array(quad_corners, dtype=np.int32)
    return cv2.pointPolygonTest(pts, (float(mid_point[0]), float(mid_point[1])), False) >= 0

def find_current_past_position(img_1, img_2, board_squares, bool_position, FEN_line, chess_board, number_to_position_map, map_position):
    past_black_bool_position = fen2board_black(FEN_line)
    diff_position = np.zeros((8, 8), dtype=int)

    image_diff = cv2.absdiff(img_1, img_2)
    image_diff_gray = cv2.cvtColor(image_diff, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(image_diff_gray, 12, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) >= 2:
        required_contours_mid_point = []
        for c in cnts:
            if cv2.contourArea(c) > 200:
                (x, y, w, h) = cv2.boundingRect(c)
                required_contours_mid_point.append([x + int(w / 2), y + int(h / 2)])

        flag = np.zeros((8, 8), dtype=int)
        
        # Iterates through dictionary keys (r, c) tuples instead of list indices
        for (r, c), quad_corners in board_squares.items():
            for mid_point in required_contours_mid_point:
                if point_in_quad(mid_point, quad_corners) and flag[r][c] == 0:
                    diff_position[r][c] = 2
                    flag[r][c] = 1

        temp_matrix = past_black_bool_position - diff_position
        position_of_past_black = np.where(temp_matrix == -1)
        position_of_new_black = np.where(temp_matrix == -2)

        if len(position_of_past_black[0]) == 0 or len(position_of_new_black[0]) == 0:
            return " ", img_2, 0

        r1, c1 = position_of_past_black[0][0], position_of_past_black[1][0]
        r2, c2 = position_of_new_black[0][0], position_of_new_black[1][0]

        player_moved = chess_board[r1][c1]
        chess_board[r1][c1] = "1"
        chess_board[r2][c2] = player_moved

        move_word = number_to_position_map[r1][c1] + number_to_position_map[r2][c2]

        draw_img = img_2.copy()
        
        # Highlight source and target squares using the dictionary coordinates
        pts1 = np.array(board_squares[(r1, c1)], np.int32).reshape((-1, 1, 2))
        pts2 = np.array(board_squares[(r2, c2)], np.int32).reshape((-1, 1, 2))
        cv2.polylines(draw_img, [pts1], True, (0, 0, 255), 2)
        cv2.polylines(draw_img, [pts2], True, (0, 255, 0), 2)

        return move_word, draw_img, 1
    else:
        return " ", img_2, 0
