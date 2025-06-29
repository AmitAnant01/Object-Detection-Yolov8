import cv2
import csv
import time
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict


# Load yolo 8 
model = YOLO("yolov8n.pt")

# Loading the video for the detecting of the object inside them
cap = cv2.VideoCapture("video.mp4")

# CSV logging setup
csv_file = open('detection_log.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Object Name", "Total Count"])
font = cv2.FONT_HERSHEY_SIMPLEX

#Creating the line position for the  detection of the objects
line_y = 350
line_offset = 10

# Object counters start time fbs
object_counters = defaultdict(int)
start_time = time.time()
frame_count = 0

# Predefined color map 
def get_color(class_name):
    color_palette = {
        "person": (0, 255, 255),
        "car": (0, 255, 0),
        # "truck": (0, 0, 255),
        # "bus": (255, 0, 0),
        "motorbike": (255, 255, 0),
        "bicycle": (255, 0, 255)
        # "dog": (160, 82, 45),
        # "cat": (255, 105, 180),
        # "bird": (100, 149, 237),
        # "horse": (139, 69, 19),
        # "botel":(140, 180,250),
        # "lion":(160,200,100)
    }
    return color_palette.get(class_name, (200, 200, 200))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    frame_count += 1

    results = model(frame, verbose=False)[0]

    for box in results.boxes.data:
        x1, y1, x2, y2, score, cls_id = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        class_id = int(cls_id)
        class_name = model.names.get(class_id, "Unknown")

        if score > 0.5:
            center_y = int((y1 + y2) / 2)

            # Count only when center of object crosses virtual line
            if abs(center_y - line_y) < line_offset:
                object_counters[class_name] += 1
                csv_writer.writerow([timestamp, class_name, object_counters[class_name]])

            color = get_color(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), font, 0.6, color, 2)

    # it will draw counting line
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 255, 0), 2)

    # Display all object counters
    cv2.rectangle(frame, (0, 0), (350, 30 + 25 * len(object_counters)), (0, 0, 0), -1)
    y_offset = 20
    for obj, count in sorted(object_counters.items()):
        cv2.putText(frame, f"{obj.capitalize()}: {count}", (10, y_offset), font, 0.7, (255, 255, 255), 2)
        y_offset += 25

    # To display the fps it will created
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 120, 25), font, 0.6, (0, 255, 255), 2)

    # showing the output while detection the object
    cv2.putText(frame, f"Time: {timestamp}", (frame.shape[1] - 250, 55), font, 0.6, (255, 255, 255), 2)

    cv2.imshow("🧠 YOLOv8 Smart Object Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
csv_file.close()