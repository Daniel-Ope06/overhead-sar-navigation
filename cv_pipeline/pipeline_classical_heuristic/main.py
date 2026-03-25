import sys
import os
import glob
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from cv_pipeline.modules.target_detector import TargetDetector  # noqa: E402, E501, I001
from cv_pipeline.modules.color_segmenter import ColorSegmenter  # noqa: E402, E501, I001
from cv_pipeline.modules.path_finder import PathFinder  # noqa: E402, I001
from cv_pipeline.modules.visualizer import Visualizer  # noqa: E402, I001

WEIGHTS_PATH = os.path.join(
    ROOT_DIR, "model_training",
    "object_detection", "weights", "maze_actor_detector_yolo26s_v1.pt")

TEST_IMAGES_DIR = os.path.join(
    ROOT_DIR, "data_generation", "synthetic_dataset", "images", "test")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test_runs")


def main():
    GRID_RESOLUTION = 30

    detector = TargetDetector(WEIGHTS_PATH, grid_size=GRID_RESOLUTION)
    segmenter = ColorSegmenter(grid_size=GRID_RESOLUTION)
    path_finder = PathFinder()
    visualizer = Visualizer(grid_size=GRID_RESOLUTION)

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png"))

    for img_path in test_images:
        filename = os.path.basename(img_path)

        targets = detector(img_path)
        ugv_data = targets.get(0)
        human_data = targets.get(1)

        start_node = ugv_data['grid_node']  # type: ignore
        goal_node = human_data['grid_node']  # type: ignore

        binary_matrix = segmenter.generate_matrix(img_path)
        optimal_path = path_finder.find_path(
            binary_matrix, start_node, goal_node)

        final_output = visualizer(
            img_path, targets, optimal_path)  # type: ignore
        save_path = os.path.join(OUTPUT_DIR, f"solved_{filename}")
        cv2.imwrite(save_path, final_output)  # type: ignore
        print(f"    -> Exported: {save_path}")

    print(f"\n[+] Processing complete. All results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
