from ultralytics import YOLO
import cv2
import cvzone
import math
import os

# Load YOLO model
model = YOLO("best_dect_model.pt")

# Video capture
cap = cv2.VideoCapture("Videos/v4.mp4")
ret, frame = cap.read()
if not ret:
    print("❌ Could not read video.")
    exit()
height, width, _ = frame.shape
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

video_writer = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'scooter', 'bicycle']
vehicle_classes = [cls.lower() for cls in vehicle_classes]

count_line_y = height // 2 + 50
vehicle_count = 0

# Track vehicles across frames (simulated tracking)
vehicle_tracks = []
counted_ids = set()
next_id = 0

def get_closest_id(cx, cy, tracks, threshold=40):
    for i, track in enumerate(tracks):
        tx, ty, tid = track
        if math.hypot(cx - tx, cy - ty) < threshold:
            return tid, i
    return None, None

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    new_tracks = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cx = x1 + w // 2
            cy = y1 + h // 2
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls].lower()

            if conf > 0.3 and class_name in vehicle_classes:
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{class_name} {conf:.2f}', (x1, max(20, y1 - 10)), 1, 1)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

                # Try to match with previous track
                match_id, match_index = get_closest_id(cx, cy, vehicle_tracks)

                if match_id is not None:
                    new_tracks.append((cx, cy, match_id))
                    if match_id not in counted_ids and count_line_y - 25 < cy < count_line_y + 25:
                        vehicle_count += 1
                        counted_ids.add(match_id)
                else:
                    # Assign new ID
                    new_tracks.append((cx, cy, next_id))
                    if count_line_y - 25 < cy < count_line_y + 25:
                        vehicle_count += 1
                        counted_ids.add(next_id)
                    next_id += 1

    vehicle_tracks = new_tracks.copy()

    cv2.line(img, (0, count_line_y), (width, count_line_y), (0, 255, 255), 2)
    cv2.putText(img, f'Vehicle Count: {vehicle_count}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    video_writer.write(img)
    cv2.imshow("Vehicle Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
print("✅ Output saved to: output_video.mp4")
