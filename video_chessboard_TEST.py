# This is a program to test the camera and detect a chessboard pattern
# It shows the camera video stream and highlights a chessboard if found


# General libraries
import time
# Libraries to control the camera
from picamera2 import Picamera2
import cv2


# -----------------------------------------------------
# Chessboard settings
# -----------------------------------------------------
# CHESSBOARD_SIZE is the number of INTERNAL corners (not squares!)
# For example, a standard 8x8-square calibration board has 7x7 internal
# corners. A common 10x7-square board has 9x6 internal corners.
# Adjust these two numbers to match your printed board.
CHESSBOARD_SIZE = (9, 6)  # (columns, rows) of internal corners

# Speed things up: chessboard detection is slow, so we don't need to run it
# on every single frame.
DETECT_EVERY_N_FRAMES = 5


# Initialize the camera
camera = Picamera2()


try:
    # Start the camera
    camera.start()
    print("To end the program, press q when hovering over a window")
    print("or press CTRL+C in the terminal.")

    frame_count = 0

    # Continuously grab camera frames
    while True:
        # Grab a frame
        img = camera.capture_array()

        #-----------------------------------------------------
        # Picam natively uses RGB, but OpenCV, which we use for manipulating
        # and displaying images, uses BGR.
        # In this example, we do an explicit conversion from RGB (picamera)
        # to BGR (opencv) in order to display the images properly.
        #-----------------------------------------------------
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        #-----------------------------------------------------
        # Chessboard detection
        #-----------------------------------------------------
        # Chessboard detection works on a grayscale image, and only needs
        # to run every few frames to keep the video feed responsive.
        frame_count += 1
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Try to find the internal corners of the chessboard.
            # found is True/False, corners holds the pixel locations if found.
            found, corners = cv2.findChessboardCorners(
                gray,
                CHESSBOARD_SIZE,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if found:
                # Refine the corner locations to sub-pixel accuracy
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Draw the detected corners on the color image
                cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, found)

                # Let the user know a board was found
                cv2.putText(
                    img, "Chessboard found!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
            else:
                cv2.putText(
                    img, "No chessboard detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )

        # Show the frame (OpenCV assumes BRG color representation)
        cv2.imshow("Camera", img)

        # The waitKey command is needed to force openCV to show the image
        # It looks for a keystroke for x ms (with x the argument) and otherwise continues
        # In this case, the program check if the user pressed 'q'
        if cv2.waitKey(1) == ord('q'):
            break


# Quit the program when the user presses CTRL + C
except KeyboardInterrupt:
    pass
finally:
    # Clean up the resources
    cv2.destroyAllWindows()
    camera.stop()
    camera.close()
