from ultralytics import YOLO
import cv2
import torch
import torchvision

# ------------------ Load Models ------------------
model1 = YOLO(r"runs/detect/train6/weights/best.pt")  # Traffic + Ambulance
model2 = YOLO(r"yolo11n.pt")                            # Pretrained COCO

# ------------------ Combine Class Names ------------------
names1 = model1.names
names2 = model2.names
offset = len(names1)
combined_names = {**names1, **{k + offset: v for k, v in names2.items()}}

# ------------------ Video Source ------------------
video_path = r"image_test\WhatsApp Video 2025-10-07 at 11.53.39.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Cannot open video.")
    exit()

# ------------------ Ensemble Detection with NMS ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results1 = model1.predict(frame, imgsz=640, verbose=False)[0]
    results2 = model2.predict(frame, imgsz=640, verbose=False)[0]

    # Collect all boxes
    all_boxes = []
    all_scores = []
    all_classes = []

    for r, model_offset in zip([results1, results2], [0, offset]):
        if r.boxes is not None:
            for box in r.boxes:
                b = box.xyxy[0].cpu()
                conf = float(box.conf)
                cls = int(box.cls) + model_offset
                all_boxes.append(b)
                all_scores.append(conf)
                all_classes.append(cls)

    if all_boxes:
        boxes_tensor = torch.stack(all_boxes)          # shape [N,4]
        scores_tensor = torch.tensor(all_scores)      # shape [N]
        classes_tensor = torch.tensor(all_classes)    # shape [N]

        # Apply NMS (Non-Maximum Suppression)
        keep = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
        boxes_tensor = boxes_tensor[keep]
        scores_tensor = scores_tensor[keep]
        classes_tensor = classes_tensor[keep]

    # Draw final boxes
    annotated = frame.copy()
    for b, cls, conf in zip(boxes_tensor, classes_tensor, scores_tensor):
        b = b.int().tolist()
        label = f"{combined_names.get(int(cls), 'obj')} {conf:.2f}"
        color = (0, 255, 0) if "ambulance" in label else (255, 0, 0)
        cv2.rectangle(annotated, (b[0], b[1]), (b[2], b[3]), color, 2)
        cv2.putText(annotated, label, (b[0], b[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLO Ensemble NMS", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("YOLO Ensemble NMS", cv2.WND_PROP_VISIBLE)<1:
        break

cap.release()
cv2.destroyAllWindows()
