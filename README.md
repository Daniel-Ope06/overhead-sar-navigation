# 🚁 Ariadne: Autonomous Air-to-Ground Optical Navigation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Blender](https://img.shields.io/badge/Blender-5.1-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Ultralytics](https://img.shields.io/badge/YOLO-Object%20Detection-00FFFF)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![U-Net](https://img.shields.io/badge/U--Net-Semantic%20Segmentation-blueviolet)

> **📄 Read the full academic report:** [Ariadne: A Synthetic Proof of Concept for Air-to-Ground Optical Navigation](docs/Ariadne_Air_to_Ground_Proof_of_Concept.pdf)

In classical mythology, the labyrinth was navigated using *"Ariadne's thread"*, a lifeline provided by an outside observer to guide a vulnerable explorer safely through a maze.

**Ariadne** is the modern digital equivalent. When GPS signals are jammed or ground sensors fail in disaster zones, robots go blind. This project utilizes a single high-altitude image from an overhead drone (UAV) to act as an "oracle." By passing this image through a deep learning pipeline, the system automatically segments the environment and draws a safe, collision-free mathematical route for a ground vehicle (UGV) to follow.

<br/>

## 💡 The Problem: Navigating the Unknown

Modern ground drones rely heavily on GPS and active sensors like LiDAR. But what happens during electronic warfare, under a thick forest canopy, or in a chaotic disaster zone? Those signals die.

Ariadne shifts the paradigm to **passive optical reconnaissance**. Instead of forcing the ground robot to blindly discover obstacles by bumping into them, an overhead drone looks down, understands the geometry of the terrain, and feeds the ground robot the perfect route before it even moves.

**Why test on a Maze?** Before deploying complex AI into messy, unpredictable real-world jungles, we have to prove the underlying math actually works. A procedural 2D maze acts as our "noiseless sandbox," allowing us to rigorously test the object detection, semantic segmentation, and pathfinding algorithms without the distractions of real-world shadows or weather.

<br/>

## 🎲 Building the Sandbox (Data Generation)

> **Relevant Files:** `data_generation` directory.

To train our deep learning models to understand geometry, we needed a flawless dataset. We procedurally generated 1,000 custom mazes using Blender's Python API, which allowed us to automatically export mathematically perfect ground-truth labels for our AI to learn from.

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
  <i>Figure 2: Simulated UAV Optical Feed (left), Perfect Ground-Truth Mask (right)</i>
</div>

<br/><br/>

## 🧩 The Ariadne Pipeline

The system translates raw pixels into physical movement through three core phases:

### 1. 🎯 Finding the Needle (Object Detection)

First, the system needs to know where we are and where we are going. A custom-trained **YOLO** (You Only Look Once) model scans the raw optical feed, successfully hunting down the exact pixel coordinates of our Ground Robot (Start) and our Human Target (Goal).

* **Input:** RGB overhead image.
* **Output:** Bounding box coordinates and confidence scores.
* **Environment:**
  * Model Training: **Colab** (`model_training/object_detection/`)
  * Inference execution: **Local** (`cv_pipeline/modules/target_detector.py`)

<br/>

<div align="center">
  <img src="showcase/object_detection/yolo_maze_env_0528.png" width="60%" alt="YOLO Target Detection">
  <br/>
  <i>Figure 3: YOLO successfully isolating the UGV and Human target.</i>
</div>

### 2. 🗺️ Seeing the Walls (Environmental Segmentation)

Next, the pipeline has to figure out what is a safe path and what is a concrete wall. We built and tested two different approaches to prove why deep learning is necessary:

* **Approach A: Classical Color Heuristics (`cv_pipeline/modules/color_segmenter.py`)**
  * Uses OpenCV to filter the image purely by color. While incredibly fast, it is easily confused by shadows and fuzzy pixel edges, making it dangerous for real robots.

* **Approach B: Neural Semantic U-Net (`cv_pipeline/modules/unet_segmenter.py`)**
  * Powered by a PyTorch U-Net architecture with a ResNet34 backbone. Instead of just looking at colors, it actually learns the structural geometry of the walls, creating a rock-solid, mathematically perfect map.

* **Environment:**
  * U-Net Training: **Colab** (`model_training/image_segmentation/unet_training.ipynb`)
  * Inference execution: **Local** (Both segmenter modules)

<br/>

<div align="center">
  <img src="showcase/image_segmentation/color_threshold_tuner.png" width="44%" alt="UNET Mask Output">
  <br/>
  <i>Figure 4: Custom built Color Threshold Tuner for Approach A</i>
</div>

<br/>

<div align="center">
  <img src="showcase/image_segmentation/maze_env_0335.png" width="44%" alt="RGB Image Input">
  &nbsp;
  <img src="showcase/image_segmentation/maze_env_0335_color.png" width="44%" alt="Color Threshold Mask Output">
  <br/>
  <i>Figure 5: RGB Input (left), Approach A Classical Heuristic Output (right). Notice the fuzzy, unreliable edges.</i>
</div>

<br/><br/>

<div align="center">
  <img src="showcase/image_segmentation/maze_env_0335.png" width="44%" alt="RGB Image Input">
  &nbsp;
  <img src="showcase/image_segmentation/maze_env_0335_unet.png" width="44%" alt="UNET Mask Output">
  <br/>
  <i>Figure 6: RGB Input (left), Approach B Neural Semantic Output (right). A perfectly rigid, traversable map.</i>
</div>

<br/><br/>

### 3. 🛤️ Drawing the Map (Tactical Pathfinding)

Finally, Ariadne takes the start/goal nodes from YOLO and the rigid map from the U-Net, transmuting them into a mathematical cost matrix. Utilizing the **A* (A-Star) search algorithm**, the system calculates the absolute shortest path while maintaining a strict physical buffer to ensure the robot never clips a wall corner.

* **Environment:** Executed **Local** (`cv_pipeline/modules/path_finder.py`)

<br/>

## 🚀 Final Results: Pixels to Physical Routes

The final output of the Ariadne pipeline is a fully autonomous, collision-free waypoint route projected seamlessly back onto the original visual space.

> 📂 Explore more routing examples and edge-cases in the `showcase` directory.

<div align="center">
  <img src="showcase/solved_examples/classical_heuristic/solved_maze_env_0335.png" width="44%" alt="Classical heuristic solution">
  &nbsp;
  <img src="showcase/solved_examples/neural_semantic/solved_maze_env_0335.png" width="44%" alt="Neural semantic solution">
  <br/>
  <i>Figure 7: Standard routing. Classical heuristic (left) vs. the highly optimized Neural U-Net path (right).</i>
</div>

<br/>

<div align="center">
  <img src="showcase/solved_examples/classical_heuristic/solved_maze_env_0134.png" width="44%" alt="Classical heuristic solution">
  &nbsp;
  <img src="showcase/solved_examples/neural_semantic/solved_maze_env_0134.png" width="44%" alt="Neural semantic solution">
  <br/>
  <i>Figure 8: Complex routing. The system successfully navigates high-density obstacles without crashing.</i>
</div>

<br/>

## 🔭 Future Directions

With the core logic of the pipeline validated in this proof-of-concept, Ariadne is structurally prepared to scale to real-world deployment:

* **Unstructured Environments (Forest Canopies):** Upgrading the U-Net training weights from synthetic mazes to complex, messy real-world topography where traditional sensors fail.
* **Dynamic Terrain Weights:** Evolving the binary map (wall vs. path) into a multi-class map (mud vs. asphalt), allowing the A* algorithm to calculate routes that are not just the shortest, but the most energy-efficient for the hardware.
* **Tactical Edge Deployment:** Optimizing the pipeline for deployment on SWaP-constrained (Size, Weight, and Power) edge accelerators, such as the NVIDIA Jetson architecture.
