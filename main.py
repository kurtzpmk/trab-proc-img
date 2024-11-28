import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from crop_img import crop_img
from resize_img import resize_img

model = YOLO('./best.pt')

def load_and_predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = resize_img(crop_img(cv2.imread(file_path)))
        results = model.predict(source=img, imgsz=640, conf=0.5)
        display_image(draw_boxes(img.copy(), results))
        display_results(results)

def draw_boxes(img, results):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls, conf = int(box.cls[0]), box.conf[0]
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return img

def display_image(img):
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    image_label.config(image=imgtk)
    image_label.image = imgtk

def display_results(results):
    result_text = '\n'.join(
        f"Classe: {model.names[int(box.cls[0])]}, Confiança: {box.conf[0]:.2f}"
        for result in results for box in result.boxes)
    results_label.config(text=result_text)

root = tk.Tk()
root.title("Predição de Imagem com YOLO")
root.geometry("1024x768")

tk.Button(root, text="Carregar Imagem", command=load_and_predict_image).pack(pady=10)
image_label = tk.Label(root)
image_label.pack()
results_label = tk.Label(root, text="", justify=tk.LEFT)
results_label.pack(pady=10)
root.mainloop()
