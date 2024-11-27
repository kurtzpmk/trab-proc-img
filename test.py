from ultralytics import YOLO

if __name__ == '__main__':
    m = YOLO('./runs/detect/train/weights/best.pt')
    results = m.predict(source='./Resized/Test/', imgsz=640, conf=0.5, save=True)