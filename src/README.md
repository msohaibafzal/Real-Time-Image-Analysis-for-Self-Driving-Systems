# Real-Time Image Analysis for Self-Driving Capabilities

This project is organized according to the structure used in the project report.

## Files

- `code_implementation.py` — Core implementation:
  - Lane highlighting
  - Directional decision making
  - Obstacle detection

- `real_time_processing.py` — Real-time processing flow:
  - Opens the video
  - Resizes each frame
  - Calls lane detection
  - Calls directional decision making
  - Calls obstacle detection
  - Displays the results

## Run

1. Install dependencies:

```bash
pip install opencv-python numpy
```

2. Open `real_time_processing.py`.

3. Change:

```python
VIDEO_PATH = "path/to/your/video.mp4"
```

to the path of your input video.

4. Run:

```bash
python real_time_processing.py
```

Press `Q` to stop the video.
