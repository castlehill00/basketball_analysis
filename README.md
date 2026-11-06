# 🏀 Basketball Video Analysis

Analyze basketball footage with automated detection of players, ball, team assignment, and more. This repository integrates object tracking, zero-shot classification, and custom keypoint detection for a fully annotated basketball game experience.

Leveraging the convenience of Roboflow for dataset management and Ultralytics' YOLO models for both training and inference, this project provides a robust framework for basketball video analysis.

Training notebooks are included to help you customize and fine-tune models to suit your specific needs, ensuring a seamless and efficient workflow.

## 🎓 Course Project Additions

This fork (branch `course-project`) is an image processing course project. Its aim is to tell who has the ball and whether each shot is worth two or three points, on real broadcast footage rather than on short single-camera clips. On top of the original repository it adds:

- **Shots** (`shot_detector/`): finds each shot attempt from the ball reaching the hoop, goes back to the shooter, and compares the take-off spot on the court with the three-point line.
- **A court homography that holds up on broadcast footage** (`camera_cut_detector/`, `tactical_view_converter/`): RANSAC with a mirror-label fix per frame, checks and robust smoothing per camera shot, and an adjustment to the painted court lines.
- **A cleaner ball track** (`trackers/ball_tracker.py`): the input stretched as in training, the longest plausible chain per camera shot, and only short gaps filled.
- **Possession** (`ball_aquisition/`): carried over short gaps within a camera shot, with a ball control share that adds up to 100%.
- **Long videos**: frames are decoded on demand and drawn one at a time, so memory no longer grows with the video.

Results on the second quarter of the 2023 FIBA World Cup final (25.8 minutes; rules and thresholds were tuned on the first quarter), compared with the original repository:

| | original | this fork |
|---|---|---|
| Other camera views that get a court drawn on them | 10.0% | 0.0% |
| Sudden court jumps per minute of main-camera footage | 137 | 0.4 |
| Main-camera frames with a court | 97.0% | 85.2% |
| Two or three points correct, on the shots found (11 of 33) | 6/11 | 11/11 |

Shot recall is the weak spot: in most missed shots the ball detector never finds the ball near the hoop. `evaluation/` holds the labels and the scripts behind these numbers, each comparing the final pipeline with every change undone in turn. The labels were made from the frames and were not checked by a second person.

The results were produced with Python 3.11, torch 2.11, ultralytics 8.4, supervision 0.25 and transformers 4.46.

---

## 📁 Table of Contents

1.  [Features](#-features)
2.  [Prerequisites](#-prerequisites)
3.  [Demo Video](#-demo-video)
4.  [Installation](#-installation)
5.  [Training the Models](#-training-the-models)
6.  [Usage](#-usage)
7.  [Project Structure](#-project-structure)
8.  [Future Work](#-future-work)
9.  [Contributing](#-contributing)
10. [License](#-license)

---

## ✨ Features

- Player and ball detection/tracking using pretrained models.
- Court keypoint detection for visualizing important zones.
- Team assignment with jersey color classification.
- Ball possession detection, pass detection, and interception detection.
- Shot detection and two/three-point classification from where the shooter took off.
- Easy stubbing to skip repeated computation for fast iteration.
- Various “drawers” to overlay detected elements onto frames.

---

## 🎮 Demo Video

Below is the final annotated output video.

[![BasketBall Analysis Demo Video](https://img.youtube.com/vi/xWpP0LjEUng/0.jpg)](https://youtu.be/xWpP0LjEUng)

## 🔧 Prerequisites

- Python 3.8+
- (Optional) Docker

---

## ⚙️ Installation

Setup your environment locally or via Docker.

### Python Environment

1. Create a virtual environment (e.g., venv/conda).
2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Docker

#### Build the Docker image:

```bash
docker build -t basketball-analysis .
```

#### Verify the image:

```bash
docker images
```

## 🎓 Training the Models

Harnessing the powerful tools offered by Roboflow and Ultralytics makes it straightforward to manage datasets, handle annotations, and train advanced object detection models. Roboflow provides an intuitive platform for dataset preprocessing and augmentation, while Ultralytics’ YOLO architectures (v5, v8, and beyond) deliver state-of-the-art detection performance.

This repository relies on trained models for detecting basketballs, players, and court keypoints. You have two options to get these models:

1. Download the Pretrained Weights

   - ball_detector_model.pt  
     (https://drive.google.com/file/d/1KejdrcEnto2AKjdgdo1U1syr5gODp6EL/view?usp=sharing)
   - court_keypoint_detector.pt  
     (https://drive.google.com/file/d/1nGoG-pUkSg4bWAUIeQ8aN6n7O1fOkXU0/view?usp=sharing)
   - player_detector.pt  
     (https://drive.google.com/file/d/1fVBLZtPy9Yu6Tf186oS4siotkioHBLHy/view?usp=sharing)

   Simply download these files and place them into the `models/` folder in your project. This allows you to run the pipelines without manually retraining.

2. Train Your Own Models  
   The training scripts are provided in the `training_notebooks/` folder. These Jupyter notebooks use Roboflow datasets and the Ultralytics YOLO frameworks to train various detection tasks:

   - `basketball_ball_training.ipynb`: Trains a basketball ball detector (using YOLOv5). Incorporates motion blur augmentations to improve ball detection accuracy on fast-moving game footage.
   - `basketball_court_keypoint_training.ipynb`: Uses YOLOv8 to detect keypoints on the court (e.g., lines, corners, key zones).
   - `basketball_player_detection_training.ipynb`: Trains a player detection model (using YOLO v11) to identify players in each frame.

   You can easily run these notebooks in Google Colab or another environment with GPU access. After training, download the newly generated `.pt` files and place them in the `models/` folder.

## Once you have your models in place, you may proceed with the usage steps described above. If you want to retrain or fine-tune for your specific dataset, remember to adjust the paths in the notebooks and in `main.py` to point to the newly generated models.

## 🚀 Usage

You can run this repository’s core functionality (analysis pipeline) with Python or Docker.

### 1) Using Python Directly

Run the main entry point with your chosen video file:

```bash
python main.py path_to_input_video.mp4 --output_video output_videos/output_result.avi --league fiba
```

- By default, intermediate “stubs” (pickled detection results) are used if found, allowing you to skip repeated detection/tracking. They are kept per video under `stubs/<video name>/`.
- Use the `--stub_path` flag to specify a custom stub folder, or disable stubs if you want to run everything fresh.
- Use `--league fiba` (also the CBA) or `--league nba` for the court dimensions and the three-point line.
- The detected shots are also saved next to the video, as `<output name>_shots.json`.

### 2) Using Docker

#### Build the container if not built already:

```bash
docker build -t basketball-analysis .
```

#### Run the container, mounting your local input video folder:

```bash
docker run \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/output_videos:/app/output_videos \
  basketball-analysis \
  python main.py videos/input_video.mp4 --output_video output_videos/output_result.avi
```

---

## 🏰 Project Structure

- `main.py`  
  – Orchestrates the entire pipeline: reading video frames, running detection/tracking, team assignment, drawing results, and saving the output video.

- `trackers/`  
  – Houses `PlayerTracker` and `BallTracker`, which use detection models to generate bounding boxes and track objects across frames.

- `utils/`  
  – Contains helper functions like `bbox_utils.py` for geometric calculations, `stubs_utils.py` for reading and saving intermediate results, and `video_utils.py` for reading/saving videos.

- `drawers/`  
  – Contains classes that overlay bounding boxes, court lines, passes, etc., onto frames.

- `ball_aquisition/`  
  – Logic for identifying which player is in possession of the ball.

- `pass_and_interception_detector/`  
  – Identifies passing events and interceptions.

- `court_keypoint_detector/`  
  – Detects lines and keypoints on the court using the specified model.

- `camera_cut_detector/`  
  – Splits a broadcast video into camera shots from the colour histograms of consecutive frames.

- `tactical_view_converter/`  
  – Estimates the homography from frame to court: fitted per frame, checked and smoothed per camera shot, and adjusted to the painted court lines. Holds the court dimensions of each league.

- `shot_detector/`  
  – Finds shot attempts and classifies them as two- or three-pointers.

- `team_assigner/`  
  – Uses zero-shot classification (Hugging Face or similar) to assign players to teams based on jersey color.

- `configs/`  
  – Holds default paths for models, stubs, and output video.

- `evaluation/`  
  – Labels and scripts that measure each stage against ground truth, with every change undone in turn.

---

## 🔮 Future Work

As we continue to enhance the capabilities of this basketball video analysis tool, several areas for future development have been identified:

1. **Integrating a Pose Model for Advanced Rule Detection**  
   Incorporating a pose detection model could enable the identification of complex basketball rules such as double dribbling and traveling. By analyzing player movements and positions, the system could automatically flag these infractions, adding another layer of analysis to the video footage.

These enhancements will further refine the analysis capabilities and provide users with more comprehensive insights into basketball games.

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear explanation of your changes.

---

## 🐜 License

This project is licensed under the MIT License.  
See `LICENSE` for details.

---

## 💬 Questions or Feedback?

Feel free to open an issue or reach out via email if you have questions about the project, suggestions for improvements, or just want to say hi!

Enjoy analyzing basketball footage with automatic detection and tracking!
