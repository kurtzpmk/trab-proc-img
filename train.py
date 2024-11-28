from ultralytics import YOLO

if __name__ == '__main__':
    m = YOLO('yolo11n.pt')
    training_results = m.train(data='brain-tumor.yaml', batch=-1, optimizer='auto', patience=10) # imgsz=640