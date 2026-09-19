import cv2
import numpy as np

from code_implementation import (
    highlight_largest_lane_region,
    directional_decision_making,
    detect_obstacles,
)

def process_video_realtime(video_path):
    """Process a video frame-by-frame in real time."""

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Failed to open video.")
        return

    # Simulated moving object
    obj = np.ones((20, 20, 3), dtype=np.uint8) * np.array(
        [0, 0, 255], dtype=np.uint8
    )

    # Initial object position
    x1, x2 = 400, 420
    y1, y2 = 250 - 10, 250 + 10

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Resize input frame
        frame = cv2.resize(frame, (500, 500))

        # 1. Highlight lane
        frame, edge_map = highlight_largest_lane_region(frame)

        # 2. Apply directional decision-making
        frame, x1, x2, y1, y2 = directional_decision_making(
            frame, obj, x1, x2, y1, y2
        )

        # 3. Detect obstacles
        frame = detect_obstacles(frame)

        # Display results
        cv2.imshow("Edge Map", edge_map)
        cv2.imshow("Lane + Obstacles", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    VIDEO_PATH = "path/to/your/video.mp4"
    process_video_realtime(VIDEO_PATH)
