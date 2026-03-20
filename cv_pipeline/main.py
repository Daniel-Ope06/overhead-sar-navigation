import os
# import numpy as np

from modules.target_detector import TargetDetector
from modules.color_segmenter import ColorSegmenter
from modules.path_finder import PathFinder


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    IMG_PATH = os.path.join(PROJECT_ROOT, "data_generation",
                            "synthetic_dataset", "images", "maze_env_0000.png")
    LBL_PATH = os.path.join(PROJECT_ROOT, "data_generation",
                            "synthetic_dataset", "labels", "maze_env_0000.txt")

    GRID_RESOLUTION = 30
    detector = TargetDetector(grid_size=GRID_RESOLUTION)
    segmenter = ColorSegmenter(grid_size=GRID_RESOLUTION)
    path_finder = PathFinder()

    print("\n--- Target Detection ---")
    try:
        targets = detector.fake_detect(IMG_PATH, LBL_PATH)

        ugv_data = targets.get(0)
        human_data = targets.get(1)

        if not ugv_data or not human_data:
            print("Error: Could not locate both UGV and Human ")
            return

        start_node = ugv_data['grid_node']
        goal_node = human_data['grid_node']

        print(f"[+] UGV located at matrix node: {start_node}")
        print(f"[+] Human located at matrix node: {goal_node}")
    except Exception as e:
        print(f"[!] Detection Error: {e}")
        return

    print("\n--- Color Segmentation ---")
    # segmenter.tune_thresholds(IMG_PATH)
    try:
        binary_matrix, original_img = segmenter.generate_matrix(IMG_PATH)
        print(f"Matrix Shape: {binary_matrix.shape}")
    except Exception as e:
        print(f"[!] Segmentation Error: {e}")
        return

    print("\n--- Path Finding ---")
    try:
        print("[*] Calculating optimal route...")
        optimal_path = path_finder.find_path(
            binary_matrix, start_node, goal_node)

        if not optimal_path:
            print("[!] ROUTING FAILED: No valid path exists.")
        else:
            print(f"Path found in {len(optimal_path)} steps!")
            print(f"(First 5 steps): {optimal_path[:5]} ...")
            print(f"(Last 5 steps): ... {optimal_path[-5:]}")
    except Exception as e:
        print(f"[!] Pathfinding Error: {e}")


if __name__ == "__main__":
    main()
