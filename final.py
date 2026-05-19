from ultralytics import YOLO
import cv2
import torch
import torchvision
import time
import requests

# ------------------ Load YOLO Models ------------------
model1 = YOLO(r"runs/detect/train6/weights/best.pt")   # Custom model
model2 = YOLO(r"yolo11n.pt")                           # Pretrained COCO

# ------------------ Combine Class Names ------------------
names1 = model1.names
names2 = model2.names
offset = len(names1)
combined_names = {**names1, **{k + offset: v for k, v in names2.items()}}

# ------------------ ESP32 Configuration ------------------
ESP32_URL = "http://10.139.87.181/update"   # Your ESP32 IP

# ------------------ Detection Function ------------------
def detect_objects(source, is_video=False):
    ambulance_detected = False
    vehicle_count = 0

    if is_video:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {source}")
            return None, 0, False
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"⚠️ Unable to read frame from video {source}")
            return None, 0, False
    else:
        frame = cv2.imread(source)
        if frame is None:
            print(f"❌ Cannot open image: {source}")
            return None, 0, False

    results1 = model1.predict(frame, imgsz=640, verbose=False)[0]
    results2 = model2.predict(frame, imgsz=640, verbose=False)[0]

    all_boxes, all_scores, all_classes = [], [], []

    for r, model_offset in zip([results1, results2], [0, offset]):
        if r.boxes is not None:
            for box in r.boxes:
                b = box.xyxy[0].cpu()
                conf = float(box.conf)
                cls = int(box.cls) + model_offset
                all_boxes.append(b)
                all_scores.append(conf)
                all_classes.append(cls)

    annotated = frame.copy()

    if all_boxes:
        boxes_tensor = torch.stack(all_boxes)
        scores_tensor = torch.tensor(all_scores)
        classes_tensor = torch.tensor(all_classes)

        keep = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
        boxes_tensor = boxes_tensor[keep]
        scores_tensor = scores_tensor[keep]
        classes_tensor = classes_tensor[keep]

        for b, cls, conf in zip(boxes_tensor, classes_tensor, scores_tensor):
            b = b.int().tolist()
            label = combined_names.get(int(cls), 'obj')
            color = (0, 255, 0) if "ambulance" in label.lower() else (255, 0, 0)

            if "ambulance" in label.lower():
                ambulance_detected = True
            elif label.lower() in ["car", "bus", "truck", "bike", "auto", "motorcycle"]:
                vehicle_count += 1

            cv2.rectangle(annotated, (b[0], b[1]), (b[2], b[3]), color, 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (b[0], b[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return annotated, vehicle_count, ambulance_detected


# ------------------ Input Sources ------------------
lanes = {
    "Lane1": r"image_test\WhatsApp Image 2025-10-13 at 14.32.28 (1).jpeg",
    "Lane2": r"image_test\WhatsApp Image 2025-10-13 at 14.31.01.jpeg",
    "Lane3": r"image_test\WhatsApp Image 2025-10-13 at 14.30.59.jpeg",
    "Lane4": r"image_test\maxresdefault.jpg"
}

# ------------------ Analyze Each Lane ------------------
lane_data = {}
for lane, path in lanes.items():
    is_video = path.lower().endswith(('.mp4', '.avi', '.mov'))
    print(f"🔍 Processing {lane} ({'video' if is_video else 'image'})...")
    annotated, count, amb = detect_objects(path, is_video=is_video)
    lane_data[lane] = {"count": count, "ambulance": amb, "image": annotated}

# ------------------ Dynamic Green Time Calculation ------------------
T_cycle = 120   # Total cycle time in seconds
T_min = 10      # Minimum green time per lane
L = len(lane_data)
total_vehicles = sum(d["count"] for d in lane_data.values())

green_times = {}
priority_lane = None

# Priority for ambulance
for lane, data in lane_data.items():
    if data["ambulance"]:
        priority_lane = lane
        green_times[lane] = 30  # Fixed 30 sec for ambulance lane

# Calculate dynamic times for other lanes
for lane, data in lane_data.items():
    if lane == priority_lane:
        continue
    N_i = data["count"]
    if total_vehicles > 0:
        T_i = T_min + (N_i / total_vehicles) * (T_cycle - L * T_min)
    else:
        T_i = T_min
    green_times[lane] = round(min(max(T_i, T_min), 60))  # clamp between 10–60s

# Reorder lanes (ambulance first if any)
lane_order = list(green_times.keys())
if priority_lane:
    lane_order.remove(priority_lane)
    lane_order.insert(0, priority_lane)

print("\n🟢 Signal Schedule (Dynamic):")
for lane in lane_order:
    print(f"{lane}: {green_times[lane]} sec")

# ------------------ Timer + ESP32 Communication ------------------
for lane in lane_order:
    timer = green_times[lane]
    print(f"\n🚦 {lane} is GREEN for {timer} seconds.")

    # Send ON signal to ESP32
    try:
        requests.get(f"{ESP32_URL}?lane={lane}&state=green", timeout=1)
    except:
        print(f"⚠️ Unable to reach ESP32 for {lane}")

    while timer > 0:
        for ln, data in lane_data.items():
            if data["image"] is None:
                continue
            display = data["image"].copy()

            status = "GREEN" if ln == lane else "RED"
            color = (0, 255, 0) if ln == lane else (0, 0, 255)
            text = f"{ln} | Veh: {data['count']} | Amb: {'Yes' if data['ambulance'] else 'No'}"
            cv2.putText(display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display, status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            if ln == lane:
                cv2.putText(display, f"⏳ {timer}s", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.imshow(ln, display)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()
        timer -= 1

    # Turn off after timer ends
    try:
        requests.get(f"{ESP32_URL}?lane={lane}&state=red", timeout=1)
    except:
        print(f"⚠️ Unable to turn off {lane}")

print("\n✅ All lanes processed successfully.")
cv2.destroyAllWindows()
