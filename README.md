# UAV Vision-Based Object Tracking and Camera Motion Estimation

A computer vision and perception pipeline for UAV footage, combining YOLO11n object detection, ByteTrack multi-object tracking, and visual odometry to estimate camera motion and orientation.

## Overview

This project explores a UAV perception system that combines object-level understanding with camera motion estimation.

The system integrates:

- **YOLO11n:** Object detection.
- **ByteTrack:** Multi-object tracking with persistent IDs.
- **ORB Feature Matching:** Visual feature extraction and correspondence.
- **Visual Odometry (VO):** Camera motion estimation.
- **Camera Orientation Estimation:** Roll, pitch and yaw extraction from estimated camera rotation.
- **Perception Fusion:** Association of tracked objects with camera pose and timestamps.

The objective is to develop a foundation for UAV perception applications such as aerial surveillance, object tracking, autonomous navigation and environmental monitoring.

## System Architecture

```text
                  UAV Video Input
                         |
              +----------+----------+
              |                     |
         YOLO11n                 ORB Features
        Detection               Feature Matching
              |                     |
          ByteTrack             Visual Odometry
              |                     |
       Object Track IDs       Camera Pose Estimation
              |                     |
              |              Rotation + Translation
              |                     |
              +----------+----------+
                         |
                 Perception Fusion
                         |
             +-----------+-----------+
             |           |           |
          Object      Camera      Orientation
          Tracking    Position    (Roll/Pitch/Yaw)
             |           |           |
             +-----------+-----------+
                         |
                Annotated UAV Video
                + Perception CSV
```

## Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Computer Vision | OpenCV |
| Object Detection | YOLO11n |
| Object Tracking | ByteTrack |
| Feature Extraction | ORB |
| Motion Estimation | Essential Matrix, PnP |
| Numerical Computing | NumPy |
| Data Analysis | Pandas |
| Visualization | Matplotlib |
| Deep Learning Framework | PyTorch / Ultralytics |
| Development | Google Colab |

## Datasets

### 1. UAV123 – Object Detection and Tracking Experiments

UAV123 was used for initial UAV object-tracking experiments.

The project adapted the dataset for the trained single-class detector and evaluated object detection and tracking performance.

**Object detection evaluation:**

| Metric | Result |
|---|---:|
| Precision | 0.763 |
| Recall | 0.592 |
| mAP@50 | 0.624 |
| mAP@50–95 | 0.404 |

**Inference performance:**

| Component | Latency |
|---|---:|
| GPU inference | 0.7 ms |
| Preprocessing | 0.4 ms |
| Postprocessing | 0.8 ms |

These are the recorded timings from the original evaluation setup, not guaranteed end-to-end deployment latency.

**Custom tracking evaluation:**

| Metric | Result |
|---|---:|
| Success AUC | 0.8658 |
| Precision@20 pixels | 0.9961 |
| Normalized Precision | 0.9990 |
| ID Switches | 1 |
| Average IoU | 0.8696 |

Note: These are results from the project's adapted UAV123 tracking evaluation, not an official UAV123 benchmark leaderboard result.

### 2. TUM RGB-D – Visual Odometry Benchmark

Dataset: `freiburg1_xyz`

Used to develop and evaluate RGB-D visual odometry.

Dataset contents:

- RGB images
- Depth images
- RGB timestamps
- Depth timestamps
- Ground-truth camera trajectory

Camera intrinsics used:

```python
K = np.array([
    [517.3, 0, 318.6],
    [0, 516.5, 255.3],
    [0, 0, 1]
])
```

Depth scale: 5000 units per metre.

This dataset was used to evaluate metric camera trajectory reconstruction independently of the UAV video integration.

### 3. VisDrone – Integrated UAV Video

A VisDrone sample video was used for the integrated detection, tracking and monocular visual odometry prototype.

Video properties:

| Property | Value |
|---|---:|
| Resolution | 1904 × 1070 |
| FPS | 15 |
| Total Frames | 196 |
| Duration | 13.07 seconds |

The sample video:

[Download VisDrone_0001.mp4](https://huggingface.co/datasets/choucsan/SIS-Bench/resolve/main/video/VisDrone/VisDrone_0001.mp4?download=true)

Official dataset repository:

https://github.com/VisDrone/VisDrone-Dataset

The current trained YOLO model contains one class:

```python
{0: 'tracked_target'}
```

## Implementation Details

### 1. YOLO11n Object Detection

YOLO11n is used to detect the trained target class in UAV footage.

The model produces:

- Bounding box coordinates
- Confidence scores
- Class predictions

### 2. ByteTrack Multi-Object Tracking

ByteTrack associates detections across consecutive frames and assigns persistent tracking IDs.

This provides temporal object tracking rather than independent frame-by-frame detection.

The tracking output includes:

- Object bounding boxes
- Track IDs
- Number of tracked objects per frame
- Object tracking history

### 3. RGB-D Visual Odometry – Baseline V1

The first visual odometry implementation uses ORB feature matching and RGB-D PnP.

Processing pipeline:

1. Extract ORB keypoints and descriptors.
2. Match features between consecutive frames.
3. Apply Lowe's ratio test.
4. Retrieve depth values for matched image coordinates.
5. Convert valid depth observations into 3D points.
6. Estimate camera motion using PnP RANSAC.
7. Invert the relative camera transformation.
8. Accumulate camera poses to reconstruct the trajectory.

The transformation is accumulated using:

```python
T_world_camera = (
    T_world_camera @ np.linalg.inv(T_current_previous)
)
```

**V1 full-sequence evaluation:**

| Metric | Result |
|---|---:|
| RGB Frames | 798 |
| Successful Poses | 797 |
| Failed Frames | 0 |
| Average PnP Inliers | 1076.69 |
| Average Translation / Frame | 0.01043 m |
| Estimated Path Length | 8.314 m |
| ATE RMSE | 0.03370 m |
| ATE Mean | 0.02964 m |
| ATE Median | 0.02745 m |
| ATE Maximum | 0.08593 m |

**Absolute Trajectory Error (ATE):**

- RMSE: 3.37 cm
- Mean: 2.96 cm
- Median: 2.75 cm
- Maximum: 8.59 cm

ATE was calculated after timestamp association and rigid trajectory alignment without scale correction.

### 4. V1 Relative Pose Error

The following results were obtained using the project's 6-DoF relative pose evaluation.

| Frame Interval | Translation RMSE | Rotation RMSE |
|---|---:|---:|
| 1 frame | 0.00604 m | 0.4052° |
| 10 frames | 0.01731 m | 1.0444° |
| 30 frames | 0.03382 m | 2.0037° |

These results measure relative camera-motion consistency over different frame intervals. They use the project's custom evaluation protocol and should not be presented as official TUM benchmark scores.

### 5. Visual Odometry Experiments

Three implementations were investigated.

| Version | Main Approach | ATE RMSE |
|---|---|---:|
| V1 | Original ORB + RGB-D PnP | **0.03370 m** |
| V2 | Mutual ORB matching + PnP refinement | 0.04444 m |
| V3 | Spatially distributed feature matching | 0.06160 m |

**V2 evaluation:**

| Metric | Result |
|---|---:|
| Average Mutual Matches | 1500.19 |
| Average Refined Inliers | 1309.88 |
| Average Reprojection Error | 1.068 px |
| Estimated Path Length | 8.115 m |
| ATE RMSE | 0.04444 m |

V2 produced slightly better local relative pose errors at shorter intervals, but its global trajectory accuracy was worse than V1.

**V3 evaluation:**

| Metric | Result |
|---|---:|
| ATE RMSE | 0.06160 m |
| ATE Mean | 0.05610 m |
| ATE Median | 0.05431 m |
| ATE Maximum | 0.13355 m |

Spatial feature selection did not improve the global trajectory reconstruction in this experiment.

**Final decision:** V1 was retained as the primary RGB-D visual odometry implementation because it achieved the best ATE RMSE.

### 6. Monocular Visual Odometry for UAV Video

The VisDrone sample video does not contain depth information.

Therefore, the integration prototype uses monocular visual odometry based on:

1. ORB feature extraction.
2. Lowe's ratio test.
3. Essential Matrix estimation using RANSAC.
4. Relative rotation and translation direction estimation using `recoverPose()`.
5. Camera pose accumulation.
6. Rotation conversion into Euler angles.

The approximate camera matrix used for the initial experiment was:

```python
fx = width * 0.8
fy = width * 0.8
cx = width / 2
cy = height / 2
```

**Important limitation:** Monocular translation is recovered only up to an unknown scale. The resulting trajectory is expressed in arbitrary relative units and cannot be directly compared with the metric RGB-D trajectory.

The approximate camera intrinsics also need to be replaced with calibrated camera parameters for reliable orientation estimates.

### 7. Camera Orientation Estimation

The rotation component of the accumulated camera pose is converted into Euler angles.

```python
euler = Rotation.from_matrix(
    T_world_camera[:3, :3]
).as_euler("xyz", degrees=True)
```

The output contains:

- Roll – rotation around the X-axis.
- Pitch – rotation around the Y-axis.
- Yaw – rotation around the Z-axis.

These values represent camera orientation relative to the initial camera coordinate frame.

They are not necessarily equivalent to true drone attitude relative to gravity. That requires a reliable world reference, calibrated camera model and preferably IMU measurements.

## Integrated UAV Perception Output

The final prototype combines:

- YOLO11n bounding boxes
- ByteTrack object IDs
- Monocular camera trajectory
- Camera roll, pitch and yaw
- Timestamp-based perception records

The annotated output video displays object tracking and estimated camera orientation together.

The perception CSV contains:

| Field | Description |
|---|---|
| frame | Video frame index |
| timestamp | Frame timestamp |
| camera_x | Estimated camera X position |
| camera_y | Estimated camera Y position |
| camera_z | Estimated camera Z position |
| roll_deg | Camera roll |
| pitch_deg | Camera pitch |
| yaw_deg | Camera yaw |
| track_ids | Tracked object IDs |

## Installation

Install the required libraries:

```bash
pip install ultralytics opencv-python numpy pandas matplotlib scipy tqdm
```

## Running the Project

1. Open the notebook in Google Colab.
2. Mount Google Drive.
3. Load the trained YOLO11n model.
4. Download or upload the UAV video.
5. Run YOLO detection and ByteTrack.
6. Execute monocular visual odometry.
7. Extract camera orientation.
8. Generate the integrated annotated video.
9. Export perception results to CSV.

Example model loading:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

print(model.names)
```

## Current Limitations

1. The trained YOLO model currently supports only the `tracked_target` class.
2. Monocular VO does not provide metric translation without an additional scale source.
3. Camera intrinsics for the VisDrone video are approximated rather than calibrated.
4. Camera orientation is relative to the initial camera frame, not gravity-referenced drone attitude.
5. The integrated prototype has not yet undergone quantitative VO evaluation on the VisDrone sequence.
6. Detection and VO were initially run in separate passes. A final synchronized single-pass implementation is needed to guarantee identical tracking IDs and pose timestamps.
7. The reported RGB-D ATE scores belong to the TUM Freiburg1 benchmark and must not be attributed to the VisDrone video.

## Future Improvements

- Calibrate the UAV camera using actual camera calibration data.
- Integrate IMU measurements for gravity-referenced orientation.
- Improve monocular VO using feature filtering, robust estimation and scale recovery.
- Integrate detection and tracking with VO in a single synchronized processing pipeline.
- Add multi-class UAV object detection.
- Introduce depth estimation or stereo-based motion estimation.
- Investigate object-motion compensation and dynamic feature rejection.
- Benchmark end-to-end FPS, latency, memory consumption and CPU/GPU utilization.
- Optimize the perception pipeline for edge deployment on NVIDIA Jetson or Raspberry Pi hardware.

## Project Highlights

- Implemented YOLO-based UAV object detection.
- Integrated ByteTrack for multi-object tracking.
- Developed RGB-D visual odometry using ORB and PnP.
- Evaluated multiple VO implementations using ATE and relative pose error.
- Achieved 3.37 cm ATE RMSE on the TUM Freiburg1 RGB-D sequence.
- Implemented monocular visual odometry for UAV footage.
- Extracted camera roll, pitch and yaw from estimated rotations.
- Developed a prototype combining object tracking and camera motion information.

## References

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [ByteTrack – Multi-Object Tracking](https://github.com/ifzhang/ByteTrack)
- [OpenCV](https://opencv.org/)
- [TUM RGB-D Benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
- [UAV123 Dataset](https://cemse.kaust.edu.sa/ivul/uav123)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [ORB Feature Detection](https://docs.opencv.org/)

---

**Project status:** Integrated research prototype. RGB-D visual odometry has been quantitatively evaluated on TUM Freiburg1; the monocular UAV perception integration is a functional prototype awaiting quantitative evaluation on the UAV video.
