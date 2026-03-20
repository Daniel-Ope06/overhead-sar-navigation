import os


class TargetDetector:
    """.
    Identifies the UGV and Human targets within the maze.
    """

    def __init__(self, img_size=640, grid_size=10):
        self.img_size = img_size
        self.grid_size = grid_size
        self.cell_size = img_size // grid_size

    def fake_detect(self, image_path, label_path):
        """
        Simulates object detection by parsing YOLO format text files.
        Note: 'image_path' is included to maintain a consistent API
        for future ML integration.
        """
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file missing: {label_path}")

        # Dictionary to hold target data. Class 0: UGV, Class 1: Human
        targets = {}

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])

                # Convert normalized coordinates (0-1)
                # to absolute pixels (0-640)
                px_x = int(x_center_norm * self.img_size)
                px_y = int(y_center_norm * self.img_size)
                px_w = int(width_norm * self.img_size)
                px_h = int(height_norm * self.img_size)

                # Calculate Top-Left corner for OpenCV bounding box drawing
                top_left_x = int(px_x - (px_w / 2))
                top_left_y = int(px_y - (px_h / 2))

                # Convert pixel centers to 10x10 grid indices
                # for A* pathfinding
                grid_x = int(px_x // self.cell_size)
                grid_y = int(px_y // self.cell_size)

                # Clamp values to ensure they never exceed
                # the 0-9 grid index bounds
                grid_x = max(0, min(self.grid_size - 1, grid_x))
                grid_y = max(0, min(self.grid_size - 1, grid_y))

                # Store all spatial data required by downstream modules
                targets[class_id] = {
                    "bbox_top_left": (top_left_x, top_left_y),
                    "bbox_size": (px_w, px_h),
                    "grid_node": (grid_x, grid_y)
                }

        return targets
