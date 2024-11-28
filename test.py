from ultralytics import YOLO

if __name__ == '__main__':
    m = YOLO('./best.pt')
    results = m.predict(source='./Resized/Validation/images/', conf=0.5, save=True) # imgsz=640