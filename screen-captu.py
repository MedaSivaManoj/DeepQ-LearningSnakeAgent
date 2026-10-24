import cv2
import numpy as np
import mss
import time

# Hardcoded ROI for the Google Snake game
# ROI = {"top": 150, "left": 1000, "width": 100, "height": 50}
ROI = {"top": 150, "left": 600, "width": 100, "height": 80}

with mss.mss() as sct:
    while True:
        # Grab the region
        img = np.array(sct.grab(ROI))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        time.sleep(5)
        print(frame.tolist())  # Print pixel values for debugging
        exit(0)

        # Show live capture
        cv2.imshow("Snake Game Capture", frame)

        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
