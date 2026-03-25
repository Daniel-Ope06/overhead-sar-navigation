import os
from ultralytics import YOLO


class TargetDetector:
    """.
    Identifies the UGV and Human targets within the maze.
    """

    def __init__(self, weights_path, img_size=640, grid_size=30):
        self.img_size = img_size
        self.grid_size = grid_size
        self.cell_size = img_size / grid_size

        # Initialize the detection model
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"[!] Model weights missing: {weights_path}")
        self.model = YOLO(weights_path)

    def __call__(self, image_path):
        """
        Performs object detection on an image.
        Returns the pixel coordinates and bounding boxes of both actors
        """
        # Dictionary to hold target data. Class 0: UGV, Class 1: Human
        targets = {}

        # Run inference
        results = self.model.predict(
            source=image_path, save=False, verbose=False)
        boxes = results[0].boxes

        for box in boxes:  # type: ignore
            # Extract bounding box edges and class ID from the tensor
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])

            # Calculate dimensions
            px_w = x2 - x1
            px_h = y2 - y1

            # Calculate exact center pixel
            px_x = x1 + (px_w // 2)
            px_y = y1 + (px_h // 2)

            # Convert pixel centers to grid indices for pathfinding
            grid_x = int(px_x // self.cell_size)
            grid_y = int(px_y // self.cell_size)

            # Clamp values to ensure they never exceed grid bounds
            grid_x = max(0, min(self.grid_size - 1, grid_x))
            grid_y = max(0, min(self.grid_size - 1, grid_y))

            # Store all spatial data required by other modules
            targets[class_id] = {
                "bbox_top_left": (x1, y1),
                "bbox_size": (px_w, px_h),
                "grid_node": (grid_x, grid_y),
            }

        return targets
