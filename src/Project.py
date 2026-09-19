import cv2
import numpy as np
from skimage.feature import local_binary_pattern

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# ------------------ Lane Detection via Canny + CCA ------------------
def highlight_largest_lane_region(frame, edge_pixel_thresh=500):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    inv = cv2.bitwise_not(closed_edges)

    height, width = inv.shape
    bottom = inv[height // 2:, :]
    _, binary = cv2.threshold(bottom, 127, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = 1000  # adjusted for 500x500 scale
    best_label = -1
    best_score = 0

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        component_mask = (labels == i).astype(np.uint8) * 255
        edge_pixels_in_component = cv2.bitwise_and(bottom, bottom, mask=component_mask)
        nonzero_edge_count = cv2.countNonZero(edge_pixels_in_component)
        if nonzero_edge_count < edge_pixel_thresh:
            continue
        cx, cy = centroids[i]
        distance_from_center = abs((width / 2) - cx)
        vertical_position_score = h
        score = area - 5 * distance_from_center + vertical_position_score
        if score > best_score:
            best_score = score
            best_label = i

    if best_label == -1:
        return frame, edges

    mask = np.zeros_like(binary)
    mask[labels == best_label] = 255
    full_mask = np.zeros_like(gray)
    full_mask[height // 2:, :] = mask

    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = frame.copy()
    for cnt in contours:
        cv2.drawContours(overlay, [cnt], -1, (0, 255, 255), -1)  # filled yellow
    return overlay, edges

# ------------------ Directional Logic for Moving Object ------------------
def directional_decision_making(road,object,x1,x2, y1,y2):
    road[x1:x2, y1:y2] = object
    check_upward=True
    check_right=True
    check_left=True
    check_backward=True
    for i in range (y1,y2):                                                             ### check for forward
        if [road[x1-1,i][0],road[x1-1,i][1],road[x1-1,i][2]]==[0,255,255]:        ### move upwards
            pass
        else:
            check_upward=False
    
    if (check_upward==True):
        x1-=1
        x2-=1
    else:
        for i in range (x1,x2):                                                             ### check for right
            if [road[i,y2+1][0],road[i,y2+1][1],road[i,y2+1][2]]==[0,255,255]:        ### move right
                pass
            else:
                check_right=False
        
        if (check_right==True):
            y1+=1
            y2+=1
        else:            
            for i in range (x1,x2):                                                             ### check for left
                if [road[i,y1-1][0],road[i,y1-1][1],road[i,y1-1][2]]==[0,255,255]:        ### move left
                    pass
                else:
                    check_left=False
            
            if (check_left==True):
                y1-=1
                y2-=1

            else: 
                for i in range (y1,y2):                                                             ### check for backward
                    if [road[x1+1,i][0],road[x1+1,i][1],road[x1+1,i][2]]==[0,255,255]:        ### move backward
                        pass
                    else:
                        check_backward=False
                
                if (check_backward==True):
                    x1+=1
                    x2+=1
    return road,x1,x2, y1,y2

# ------------------ Obstacle Detection Using HOG ------------------
def detect_obstacles(frame):
    fgmask = fgbg.apply(frame)
    fgmask[fgmask == 127] = 0  # Remove shadows
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)

    contours_fg, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_fg:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w)
        area = cv2.contourArea(cnt)

        # Reduced sensitivity: detect only taller, larger objects
        if 1.2 < aspect_ratio < 4.5 and area > 1500 and h > 60:
            pad_top = 20
            pad_bottom = 10
            y_new = max(y - pad_top, 0)
            h_new = h + pad_top + pad_bottom
            cv2.rectangle(frame, (x, y_new), (x + w, y_new + h_new), (0, 0, 255), 2)
            cv2.putText(frame, "Object", (x, y_new - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --------------------------------
    # OBJECT DETECTION via Road ROI Analysis (Static Obstacles)
    # --------------------------------
    road_roi = frame[300:480, 100:540]  # Bottom central region of frame

    gray = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    blurred = cv2.GaussianBlur(binary, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 150)  # Less sensitive to small edges

    contours_obs, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_obs:
        area = cv2.contourArea(cnt)
        if area > 600:  # Reduced sensitivity: ignore small debris
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x + 100, y + 300), (x + w + 100, y + h + 300), (0, 0, 255), 2)
            cv2.putText(frame, "Object", (x + 100, y + 295), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame

# ------------------ Video Processing Pipeline ------------------
def process_video_realtime(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video.")
        return

    obj = np.ones((20, 20, 3), dtype=np.uint8) * np.array([0, 0, 255], dtype=np.uint8)
    x1, x2 = 400, 420
    y1, y2 = 250 - 10, 250 + 10  # center of 500x500 frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (500, 500))
        result, edge_map = highlight_largest_lane_region(frame)
        result, x1, x2, y1, y2 = directional_decision_making(result, obj, x1, x2, y1, y2)
        result = detect_obstacles(result)

        cv2.imshow("Edge Map", edge_map)
        cv2.imshow("Lane + Obstacles", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------ RUN ------------------
process_video_realtime('')
