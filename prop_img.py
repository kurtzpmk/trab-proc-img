import os
import cv2

def crop_img(img):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(cv2.erode(thresh, None, iterations=2), None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, img.shape[1])
    y2 = min(y + h, img.shape[0])
    return img[y1:y2, x1:x2]

def main():
    images_dir = './Dataset/Test/'
    new_images_dir = './Cropped/Test/'
    os.makedirs(new_images_dir, exist_ok=True)
    for f in os.listdir(images_dir):
        img_path = os.path.join(images_dir, f)
        if os.path.isfile(img_path):
            cv2.imwrite(os.path.join(new_images_dir, f), crop_img(cv2.imread(img_path)))

if __name__ == '__main__': main()