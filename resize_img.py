import os
import cv2

def resize_img(img, size=(640, 640)):
    return cv2.resize(img, size)

def main():
    images_dir = './Cropped/Test/'
    output_images_dir = './Resized/Test/'
    os.makedirs(output_images_dir, exist_ok=True)
    for f in os.listdir(images_dir):
        img_path = os.path.join(images_dir, f)
        if os.path.isfile(img_path) and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(img_path)
            if img is not None:
                cv2.imwrite(os.path.join(output_images_dir, f), resize_img(img))

if __name__ == '__main__': main()