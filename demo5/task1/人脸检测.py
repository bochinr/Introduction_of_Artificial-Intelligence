import os
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


# 获取 OpenCV 自带的 Haar 级联文件路径
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

# 加载分类器
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
# -------------------- 配置文件部分 --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "haar_cascade": os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml"),
    "eye_cascade": os.path.join(BASE_DIR, "models", "haarcascade_eye.xml")
}

FACE_DETECTION_SCALE = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = (30, 30)


# -------------------- 人脸处理模块 --------------------
class FaceDetector:
    def __init__(self):
        try:
            self.face_cascade = cv2.CascadeClassifier(MODEL_PATHS["haar_cascade"])
            self.eye_cascade = cv2.CascadeClassifier(MODEL_PATHS["eye_cascade"])
        except Exception as e:
            print(f"Error loading Haar cascade: {e}")
            print("Falling back to OpenCV's default path.")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECTION_SCALE,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_SIZE
        )
        return faces

    def detect_eyes(self, image, face_box):
        x, y, w, h = face_box
        face_roi = image[y:y + h, x:x + w]
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray)
        # 将眼睛坐标转换为相对于原图的坐标
        eyes = [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes]
        return eyes

    @staticmethod
    def annotate_image(image, faces, eyes):
        annotated = image.copy()
        # 绘制人脸框和坐标
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, f"Face: ({x}, {y})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 绘制眼睛框和坐标
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(annotated, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            cv2.putText(annotated, f"Eye: ({ex}, {ey})", (ex, ey - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return annotated


# -------------------- GUI模块 --------------------
class FaceDetectionGUI:
    def __init__(self, master, detector):
        self.master = master
        self.detector = detector
        self.image_path = None
        self.create_widgets()

    def create_widgets(self):
        self.master.title("人脸检测系统")

        # 图片显示区域
        self.img_label = tk.Label(self.master)
        self.img_label.pack(pady=10)

        # 上传按钮
        tk.Button(self.master, text="上传图片", command=self.upload_image).pack(pady=5)

        # 检测按钮
        tk.Button(self.master, text="检测人脸和眼睛", command=self.run_detection).pack(pady=5)

        # 坐标信息显示区域
        self.coord_text = tk.Text(self.master, height=10, width=50)
        self.coord_text.pack(pady=10)

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("图片文件", "*.jpg;*.jpeg;*.png")])
        if path:
            self.image_path = path
            self.show_thumbnail(path)
            self.coord_text.delete(1.0, tk.END)  # 清除之前的坐标信息

    def show_thumbnail(self, path):
        img = Image.open(path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk

    def run_detection(self):
        if not self.image_path:
            self.show_error("请先上传图片")
            return

        try:
            image = cv2.imread(self.image_path)
            faces = self.detector.detect_faces(image)

            eyes = []
            if len(faces) > 0:
                eyes = self.detector.detect_eyes(image, faces[0])

            # 显示坐标信息
            self.coord_text.delete(1.0, tk.END)
            for i, (x, y, w, h) in enumerate(faces):
                self.coord_text.insert(tk.END, f"人脸 {i + 1} 坐标:\n")
                self.coord_text.insert(tk.END, f"左上角: ({x}, {y})\n")
                self.coord_text.insert(tk.END, f"右下角: ({x + w}, {y + h})\n\n")

            for i, (ex, ey, ew, eh) in enumerate(eyes):
                self.coord_text.insert(tk.END, f"眼睛 {i + 1} 坐标:\n")
                self.coord_text.insert(tk.END, f"左上角: ({ex}, {ey})\n")
                self.coord_text.insert(tk.END, f"右下角: ({ex + ew}, {ey + eh})\n\n")

            # 显示标注结果
            annotated = self.detector.annotate_image(image, faces, eyes)
            cv2.imshow("检测结果", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            self.show_error(f"处理出错: {str(e)}")

    def show_error(self, message):
        error_window = tk.Toplevel()
        error_window.title("错误")
        tk.Label(error_window, text=message, fg="red").pack(padx=20, pady=20)


# -------------------- 主程序 --------------------
def main():
    detector = FaceDetector()
    root = tk.Tk()
    app = FaceDetectionGUI(root, detector)
    root.mainloop()


if __name__ == "__main__":
    main()