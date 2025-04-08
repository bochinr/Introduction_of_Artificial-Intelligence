import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- 配置文件部分 --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "haar_cascade": os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml"),
    "feature_extractor": os.path.join(BASE_DIR, "models", "nn4.small2.v1.t7")
}

THRESHOLD = 0.6
FACE_DETECTION_SCALE = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = (30, 30)


# -------------------- 人脸处理模块 --------------------
class FaceProcessor:
    def __init__(self):
        try:
            self.face_cascade = cv2.CascadeClassifier(MODEL_PATHS["haar_cascade"])
        except Exception as e:
            print(f"Error loading Haar cascade: {e}")
            print("Falling back to OpenCV's default path.")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECTION_SCALE,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_SIZE
        )
        return faces

    @staticmethod
    def align_face(image, face_box):
        x, y, w, h = face_box
        face_roi = image[y:y + h, x:x + w]
        return cv2.resize(face_roi, (96, 96))

    @staticmethod
    def annotate_image(image, faces):
        annotated = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, f"({x}, {y})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return annotated


# -------------------- 特征提取模块 --------------------
class FeatureExtractor:
    def __init__(self):
        self.net = cv2.dnn.readNetFromTorch(MODEL_PATHS["feature_extractor"])

    def get_features(self, image):
        blob = cv2.dnn.blobFromImage(
            image,
            1.0 / 255,
            (96, 96),
            (0, 0, 0),
            swapRB=True,
            crop=False
        )
        self.net.setInput(blob)
        return self.net.forward().flatten()

    @staticmethod
    def compare_features(vec1, vec2):
        return cosine_similarity([vec1], [vec2])[0][0]


# -------------------- GUI模块 --------------------
class FaceVerificationGUI:
    def __init__(self, master, processor, extractor):
        self.master = master
        self.processor = processor
        self.extractor = extractor
        self.image_paths = [None, None]
        self.image_labels = [None, None]
        self.create_widgets()

    def create_widgets(self):
        self.master.title("人脸验证系统")

        frame1 = tk.Frame(self.master)
        frame1.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(frame1, text="图片1").pack()
        self.img_label1 = tk.Label(frame1)
        self.img_label1.pack()
        tk.Button(frame1, text="上传图片", command=lambda: self.upload_image(0)).pack()

        frame2 = tk.Frame(self.master)
        frame2.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(frame2, text="图片2").pack()
        self.img_label2 = tk.Label(frame2)
        self.img_label2.pack()
        tk.Button(frame2, text="上传图片", command=lambda: self.upload_image(1)).pack()

        tk.Button(self.master, text="开始验证", command=self.run_verification).pack(pady=20)

    def upload_image(self, index):
        path = filedialog.askopenfilename(filetypes=[("图片文件", "*.jpg;*.jpeg;*.png")])
        if path:
            self.image_paths[index] = path
            self.show_thumbnail(index, path)

    def show_thumbnail(self, index, path):
        img = Image.open(path)
        img.thumbnail((150, 150))
        img_tk = ImageTk.PhotoImage(img)
        if index == 0:
            self.img_label1.config(image=img_tk)
            self.img_label1.image = img_tk
        else:
            self.img_label2.config(image=img_tk)
            self.img_label2.image = img_tk

    def run_verification(self):
        if None in self.image_paths:
            self.show_error("请先上传两张图片")
            return

        try:
            img1 = cv2.imread(self.image_paths[0])
            img2 = cv2.imread(self.image_paths[1])
            faces1 = self.processor.detect_faces(img1)
            faces2 = self.processor.detect_faces(img2)

            self.show_detection_result(img1, faces1, "图片1检测结果")
            self.show_detection_result(img2, faces2, "图片2检测结果")

            if len(faces1) == 0 or len(faces2) == 0:
                self.show_error("至少一张图片未检测到人脸")
                return

            aligned1 = self.processor.align_face(img1, faces1[0])
            aligned2 = self.processor.align_face(img2, faces2[0])
            vec1 = self.extractor.get_features(aligned1)
            vec2 = self.extractor.get_features(aligned2)

            similarity = self.extractor.compare_features(vec1, vec2)
            result = "这两张照片有很大概率是同一个人" if similarity > THRESHOLD else "有很大概率不是同一个人"
            self.show_result(f"相似度: {similarity:.2f}\n验证结果: {result}")

        except Exception as e:
            self.show_error(f"处理出错: {str(e)}")

    def show_detection_result(self, image, faces, title):
        annotated = self.processor.annotate_image(image, faces)
        cv2.imshow(title, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_result(self, message):
        result_window = tk.Toplevel()
        result_window.title("验证结果")
        tk.Label(result_window, text=message, font=("Arial", 14)).pack(padx=20, pady=20)

    def show_error(self, message):
        error_window = tk.Toplevel()
        error_window.title("错误")
        tk.Label(error_window, text=message, fg="red").pack(padx=20, pady=20)


# -------------------- 主程序 --------------------
def main():
    processor = FaceProcessor()
    extractor = FeatureExtractor()
    root = tk.Tk()
    app = FaceVerificationGUI(root, processor, extractor)
    root.mainloop()


if __name__ == "__main__":
    main()