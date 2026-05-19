from ultralytics import YOLO

if __name__ == "__main__":
    # Load the pretrained YOLO11n model
    model = YOLO("yolo11n.pt")

    # Train using the YAML file
    model.train(
        data="data.yaml",   # path to your YAML
        epochs=50,
        imgsz=640,
        batch=8,
        device=0
    )