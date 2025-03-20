# SuperGlue with RealSense Camera

This repository demonstrates how to use the SuperGlue feature matching algorithm with an Intel RealSense camera for real-time feature matching and visualization. The project is built using Python, OpenCV, PyTorch, and the RealSense SDK.

SuperGlue is a deep learning-based feature matching algorithm designed for accurate and robust keypoint matching between images. It builds upon SuperPoint, a self-supervised keypoint detector, and employs a graph neural network (GNN) to establish correspondences by considering spatial relationships and contextual information between keypoints. Unlike traditional feature matching methods that rely solely on descriptor similarity, SuperGlue enhances matches by leveraging attention mechanisms, making it highly effective for tasks like SLAM, structure-from-motion, and real-time tracking. Its ability to integrate with GPU acceleration via PyTorch allows for fast inference, making it a powerful tool for applications requiring high-precision feature matching.

![image](https://github.com/user-attachments/assets/d55d7646-43c2-46ae-b6e4-77c33b43aa25)


## Features
- **Real-Time Feature Matching**: Match keypoints between consecutive frames from a RealSense camera.
- **Customizable Visualization**: Adjust keypoint colors, matching lines, and text overlays.
- **CUDA Support**: Leverage GPU acceleration for faster inference (if available).
- **RealSense Integration**: Seamlessly integrate with Intel RealSense cameras for live video input.


https://github.com/user-attachments/assets/cdb636bb-9483-4683-9d57-81e889b66bae


## Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- PyTorch (`torch`)
- Matplotlib (`matplotlib`)
- NumPy (`numpy`)
- PyRealSense (`pyrealsense2`)

## Installation
### Clone the repository:
```bash
git clone https://github.com/your-username/superglue-realsense.git
cd superglue-realsense
```
### Install the required dependencies:
```bash
pip install -r requirements.txt
```
### Install the RealSense SDK (if not already installed):
Follow the official instructions: [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)

## Usage
### Running the Demo
1. Connect your RealSense camera to your system.
2. Run the script:
```bash
python app.py
```

### Command-Line Arguments
- `--output_dir`: Directory to save output frames (default: None).
- `--show_keypoints`: Display detected keypoints (default: False).
- `--no_display`: Disable the live visualization window (default: False).
- `--force_cpu`: Force the script to run on the CPU (default: False).
- `--superglue`: Choose SuperGlue weights (`indoor` or `outdoor`, default: `indoor`).
- `--max_keypoints`: Maximum number of keypoints to detect (default: -1 for no limit).
- `--keypoint_threshold`: Keypoint detection confidence threshold (default: 0.005).
- `--match_threshold`: SuperGlue match threshold (default: 0.2).

#### Example:
```bash
python app.py --output_dir ./output --show_keypoints --superglue outdoor
```

## Code Structure
- `app.py`: Main script for running the SuperGlue demo with RealSense input.
- `models/`: Contains the SuperGlue and SuperPoint models.
- `matching.py`: SuperGlue matching implementation.
- `superpoint.py`: SuperPoint keypoint detection implementation.
- `utils.py`: Utility functions for visualization and tensor operations.
- `requirements.txt`: List of dependencies.

## Customization
### Visualization
Modify visualization parameters in `app.py`:
```python
out = make_matching_plot_fast(
    last_frame, frame_gray, kpts0, kpts1, mkpts0, mkpts1, color, text,
    show_keypoints=True, keypoint_color=(0, 255, 0), colormap=cm.plasma
)
```

### Model Configuration
Modify SuperGlue and SuperPoint configurations in `app.py`:
```python
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
    },
    'superglue': {
        'weights': 'indoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}
```

## Performance
- **FPS**: Achieves ~30 FPS on a modern GPU with CUDA support.
- **CPU Mode**: Falls back to CPU mode if no GPU is available (slower but functional).

## Troubleshooting
### Common Issues
- **RealSense Camera Not Detected:**
  - Ensure the camera is properly connected and the RealSense SDK is installed.
  - Check device permissions (e.g., `/dev/video*` on Linux).
- **CUDA Out of Memory:**
  - Reduce the `max_keypoints` parameter to limit memory usage.
  - Use `--force_cpu` to run on the CPU.
- **OpenCV Errors:**
  - Ensure OpenCV is installed correctly (`pip install opencv-python`).
  - Verify that input frames are in the correct format (grayscale or BGR).

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- **SuperGlue**: Original implementation by Paul-Edouard Sarlin, Daniel DeTone, and Tomasz Malisiewicz.
- **Intel RealSense**: For providing the RealSense SDK and hardware.
- **OpenCV**: For image processing and visualization.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

Enjoy using SuperGlue with your RealSense camera! 🚀
