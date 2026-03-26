# 🚁 Overhead Search & Rescue Navigation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Blender](https://img.shields.io/badge/Blender-5.1-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Ultralytics](https://img.shields.io/badge/YOLO-Object%20Detection-00FFFF)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![U-Net](https://img.shields.io/badge/U--Net-Semantic%20Segmentation-blueviolet)

> This project focuses on creating an autonomous navigation system that works without signals or GPS.

It uses aerial images taken from a drone to help find paths on the ground. By analyzing these images, the system maps the environment and figures out a safe route. This allows an Unmanned Ground Vehicle (UGV) to navigate difficult terrain and reach its target.

<br/>

## 💡 Inspiration & Rationale

Modern drones often depend on GPS and active signals to navigate. However, in situations like electronic warfare, dense forests, or underground disaster areas, these signals can be weak or lost. This project aims to shift the focus to **passive optical reconnaissance** using drone images taken from above to help with navigation.

**Why use a Maze?** Before deploying algorithms into messy and unpredictable areas (like forests), the math behind them needs to be tested. A synthetic 2D maze creates a simple and controlled environment. It helps to isolate and check the object detection, semantic segmentation, and pathfinding parts without the distractions of real-world lighting or natural textures.

<br/>

## 🎲 Data Generation

> **Relevant Files:** `data_generation/` directory.

To train the deep learning models, a custom dataset was procedurally generated using Blender's Python API. This method provided precise labels for every part of the environment.

<div align="center">
  <img src="showcase/data_generation/blender_viewport_setup.png" width="90%" alt="Blender Viewport Setup">
  <br/>
  <i>Figure 1: Blender rendering set up</i>
</div>

<br/><br/>

<div align="center">
  <img src="showcase/data_generation/maze_env_0000_img.png" width="44%" alt="Blender Image Output">
  &nbsp;
  <img src="showcase/data_generation/maze_env_0000_mask.png" width="44%" alt="Blender Mask Output">
  <br/>
  <i>Figure 2: Image Output (left), Mask Output (right)</i>
</div>

<br/><br/>

## 🧩 Pipeline Components

The navigation system is broken down into three sequential phases.

### 1. 🎯 Target Extraction (Object Detection)

To find a route, the system needs to know where to start and where to end. A custom **YOLO** object detection model checks the overhead image. It identifies the UGV (starting point) and the human (goal) by drawing boxes around them and turns these points into grid nodes.

* **Input:** RGB overhead image.

* **Output:** Bounding box coordinates and confidence scores.

* **Environment:**
  * Model Training: **Colab** (`model_training/object_detection/`)
  * Inference execution: **Local** (`cv_pipeline/modules/target_detector.py`)

<br/>

<div align="center">
  <img src="showcase/object_detection/yolo_maze_env_0528.png" width="60%" alt="YOLO Target Detection">
  <br/>
  <i>Figure 3: YOLO object detection sample output</i>
</div>

### 2. 🗺️ Environmental Segmentation

This study looks at two different ways to map the areas of a floor that UGVs or people can walk on.

* **Approach A: Color Heuristic (`cv_pipeline/modules/color_segmenter.py`)**
  * This method uses OpenCV's color thresholding, which is efficient and quick to compute. However, it is sensitive to changes in lighting and can be affected by pixel anti-aliasing, making it less reliable.

* **Approach B: Neural Semantic (`cv_pipeline/modules/unet_segmenter.py`)**
  * This project uses a PyTorch U-Net architecture with a pre-trained ResNet34 backbone. Instead of relying only on colors, it learns the structural shape of the walls. This approach helps create accurate and consistent masks.

* **Environment:**
  * U-Net Training: **Colab** (`model_training/image_segmentation/unet_training.ipynb`)
  * Inference execution: **Local** (Both segmenter modules)

<br/>

<div align="center">
  <img src="showcase/image_segmentation/color_threshold_tuner.png" width="44%" alt="UNET Mask Output">
  <br/>
  <i>Figure 4: Color Threshold Tuner</i>
</div>

<br/>

<div align="center">
  <img src="showcase/image_segmentation/maze_env_0335.png" width="44%" alt="RGB Image Input">
  &nbsp;
  <img src="showcase/image_segmentation/maze_env_0335_color.png" width="44%" alt="Color Threshold Mask Output">
  <br/>
  <i>Figure 5: Image Input (left), Color Heuristic Output (right)</i>
</div>

<br/><br/>

<div align="center">
  <img src="showcase/image_segmentation/maze_env_0335.png" width="44%" alt="RGB Image Input">
  &nbsp;
  <img src="showcase/image_segmentation/maze_env_0335_unet.png" width="44%" alt="UNET Mask Output">
  <br/>
  <i>Figure 6: Image Input (left), Neural Semantic Output (right)</i>
</div>

<br/><br/>

### 3. 🛤️ Tactical Pathfinding

After finding the Start and Goal points and identifying the walkable areas, the A* (A-Star) search algorithm is used to determine the best path.

**Why choose A*?**A* is effective on a 2D grid because it reliably finds the shortest path while using minimal computing power.

* **Environment:** Executed **Local** (`cv_pipeline/modules/path_finder.py`)

<br/>

## 🚀 Results

The final results show the bounding boxes and A* route placed over the original RGB image.

> 📂 Explore more routing examples and diagrams in the `/showcase` directory.

<div align="center">
  <img src="showcase/solved_examples/classical_heuristic/solved_maze_env_0335.png" width="44%" alt="Classical heuristic solution">
  &nbsp;
  <img src="showcase/solved_examples/neural_semantic/solved_maze_env_0335.png" width="44%" alt="Neural semantic solution">
  <br/>
  <i>Figure 7: Classical heuristic solution (left), Neural semantic solution (right)</i>
</div>

<br/>

<div align="center">
  <img src="showcase/solved_examples/classical_heuristic/solved_maze_env_0134.png" width="44%" alt="Classical heuristic solution">
  &nbsp;
  <img src="showcase/solved_examples/neural_semantic/solved_maze_env_0134.png" width="44%" alt="Neural semantic solution">
  <br/>
  <i>Figure 8: Classical heuristic solution (left), Neural semantic solution (right)</i>
</div>

<br/>

## 🔭 Future Directions

To turn this proof-of-concept into a practical tactical module, future versions will focus on:

* **Unstructured Environments (Forest Canopies):** Upgrading the U-Net training pipeline from synthetic mazes to more complex and messy real-world landscapes, where traditional computer vision tools struggle to function.

* **Dynamic Terrain Weights:** Expanding the semantic segmentation to classify different types of terrain (e.g., mud vs. concrete) and assigning different traversal costs to the A* nodes based on the segmentation.

* **Edge Optimization:** Making the pipeline more lightweight and easier to deploy on hardware that has limited size, weight, and power, (SWaP-constrained) like the NVIDIA Jetson.
