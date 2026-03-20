import cv2
import os
import numpy as np


class ColorSegmenter:
    """
    Isolates walls using RGB color thresholds
    to generate a high-resolution (30x30) navigable binary matrix
    Includes a built-in GUI tuner for calibrating lighting setups.
    """

    def __init__(self, img_size=640, grid_size=30):
        self.img_size = img_size
        self.grid_size = grid_size
        self.cell_size = img_size / grid_size

        # Default RGB values for purely white walls.
        # Format: [Red, Green, Blue] (0-255 scale)
        self.lower_wall_rgb = np.array([200, 200, 200])
        self.upper_wall_rgb = np.array([255, 255, 255])

    def tune_thresholds(self, image_path):
        """
        Opens an interactive GUI
        to dial in the exact RGB thresholds for the walls.
        Press 'q' to save the values and close the tuner.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image missing: {image_path}")

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # type: ignore

        def nothing(x):
            pass

        cv2.namedWindow('RGB Tuner', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RGB Tuner', 400, 300)

        cv2.createTrackbar(
            'R Min', 'RGB Tuner', self.lower_wall_rgb[0], 255, nothing)
        cv2.createTrackbar(
            'G Min', 'RGB Tuner', self.lower_wall_rgb[1], 255, nothing)
        cv2.createTrackbar(
            'B Min', 'RGB Tuner', self.lower_wall_rgb[2], 255, nothing)

        cv2.createTrackbar(
            'R Max', 'RGB Tuner', self.upper_wall_rgb[0], 255, nothing)
        cv2.createTrackbar(
            'G Max', 'RGB Tuner', self.upper_wall_rgb[1], 255, nothing)
        cv2.createTrackbar(
            'B Max', 'RGB Tuner', self.upper_wall_rgb[2], 255, nothing)

        print("Adjust sliders until walls are WHITE and paths are BLACK.")
        print("Press 'q' to lock in values and close.")

        while True:
            r_min = cv2.getTrackbarPos('R Min', 'RGB Tuner')
            g_min = cv2.getTrackbarPos('G Min', 'RGB Tuner')
            b_min = cv2.getTrackbarPos('B Min', 'RGB Tuner')

            r_max = cv2.getTrackbarPos('R Max', 'RGB Tuner')
            g_max = cv2.getTrackbarPos('G Max', 'RGB Tuner')
            b_max = cv2.getTrackbarPos('B Max', 'RGB Tuner')

            lower = np.array([r_min, g_min, b_min])
            upper = np.array([r_max, g_max, b_max])

            mask = cv2.inRange(img_rgb, lower, upper)
            cv2.imshow('RGB Mask (White = Walls)', mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.lower_wall_rgb = lower
                self.upper_wall_rgb = upper
                print("\n--- NEW RGB THRESHOLDS SAVED ---")
                print(
                    "self.lower_wall_rgb = " +
                    f"np.array([{r_min}, {g_min}, {b_min}])")
                print(
                    "self.upper_wall_rgb = " +
                    f"np.array([{r_max}, {g_max}, {b_max}])")
                break

        cv2.destroyAllWindows()
