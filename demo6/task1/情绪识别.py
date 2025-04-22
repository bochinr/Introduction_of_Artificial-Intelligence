import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# -------------------- 配置部分 --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion-ferplus-8.onnx")
EMOTIONS = ["中性", "快乐", "惊讶", "悲伤", "愤怒", "厌恶", "恐惧"]  # ONNX模型的输出顺序


# -------------------- 核心检测类 --------------------
class EmotionDetector:
    def __init__(self):
        # 初始化人脸检测器
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # 加载ONNX模型
        if not os.path.exists(MODEL_PATH):
            self.download_model()
        self.net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def download_model(self):
        """自动下载模型"""
        import urllib.request
        try:
            urllib.request.urlretrieve(
                "https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
                MODEL_PATH
            )
        except Exception as e:
            messagebox.showerror("错误", f"模型下载失败: {str(e)}")

    def detect_faces(self, img):
        """检测人脸"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.1, 5)

    def analyze_emotion(self, face_roi):
        """分析情绪（返回所有情绪概率）"""
        # 预处理
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1.0 / 255.0,
            mean=[0.5],
            swapRB=False
        )

        # 预测
        self.net.setInput(blob)
        preds = self.net.forward()[0]
        probabilities = [float(p) for p in preds]  # 转换为Python float类型

        # 返回情绪字典
        return {
            emotion: prob
            for emotion, prob in zip(EMOTIONS, probabilities)
        }

    def annotate_image(self, img, faces, emotions_list):
        """标注图像（带所有情绪信息）"""
        annotated = img.copy()
        for i, ((x, y, w, h), emotions) in enumerate(zip(faces, emotions_list)):
            # 绘制人脸框
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 显示主要情绪
            main_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            cv2.putText(
                annotated, f"{main_emotion}",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2
            )

            # 在右侧显示详细情绪（避免遮挡人脸）
            text_x = x + w + 10
            for j, (emotion, prob) in enumerate(emotions.items()):
                text = f"{emotion}: {prob:.2%}"
                cv2.putText(
                    annotated, text,
                    (text_x, y + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1
                )
        return annotated


# -------------------- GUI界面 --------------------
class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.detector = EmotionDetector()
        self.setup_ui()

    def setup_ui(self):
        self.root.title("情绪识别系统")

        # 图片显示
        self.img_label = tk.Label(self.root)
        self.img_label.pack(pady=10)

        # 按钮区
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)

        tk.Button(btn_frame, text="上传图片", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="分析情绪", command=self.run_detection).pack(side=tk.LEFT, padx=5)

        # 详细信息显示
        self.text = tk.Text(self.root, height=15, width=60)
        self.text.pack(pady=10)

    def load_image(self):
        """加载图片"""
        path = filedialog.askopenfilename(filetypes=[("图片", "*.jpg *.jpeg *.png")])
        if path:
            self.image = cv2.imread(path)
            self.show_thumbnail(path)
            self.text.delete(1.0, tk.END)

    def show_thumbnail(self, path):
        """显示缩略图"""
        img = Image.open(path)
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk

    def run_detection(self):
        """执行情绪分析"""
        if not hasattr(self, 'image'):
            messagebox.showerror("错误", "请先上传图片")
            return

        faces = self.detector.detect_faces(self.image)
        if len(faces) == 0:
            messagebox.showinfo("提示", "未检测到人脸")
            return

        # 分析每张脸的情绪
        emotions_list = []
        for (x, y, w, h) in faces:
            face_roi = self.image[y:y + h, x:x + w]
            emotions = self.detector.analyze_emotion(face_roi)
            emotions_list.append(emotions)

            # 在文本框中显示详细结果
            self.text.insert(tk.END, f"人脸 {len(emotions_list)} 情绪分析:\n")
            for emotion, prob in emotions.items():
                self.text.insert(tk.END, f"  {emotion}: {prob:.2%}\n")
            self.text.insert(tk.END, "\n")

        # 显示标注后的图片
        annotated = self.detector.annotate_image(self.image, faces, emotions_list)
        cv2.imshow("情绪分析结果", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# -------------------- 主程序 --------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()