import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import time
import subprocess
import urllib.request
import math
from collections import deque

# -------------------- 配置部分 --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "facial_expression_model.onnx")
EMOTIONS = ["愤怒", "厌恶", "恐惧", "快乐", "悲伤", "惊讶", "中性"]  # DeepFace模型的输出顺序

# 下载地址
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
DEEPFACE_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/facial_expression_recognition/facial_expression_recognition.onnx"


# -------------------- ADB工具类 --------------------
class ADBHelper:
    @staticmethod
    def check_adb_installed():
        """检查ADB是否安装"""
        try:
            subprocess.run(["adb", "--version"], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @staticmethod
    def get_connected_devices():
        """获取已连接的设备列表"""
        try:
            result = subprocess.run(["adb", "devices"], check=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)
            devices = []
            lines = result.stdout.splitlines()
            for line in lines[1:]:  # 跳过第一行标题
                if line.strip() and "device" in line:
                    devices.append(line.split("\t")[0])
            return devices
        except subprocess.CalledProcessError as e:
            print(f"ADB错误: {e.stderr}")
            return []

    @staticmethod
    def setup_forward(device_id, port=8080):
        """设置端口转发"""
        try:
            subprocess.run(["adb", "-s", device_id, "forward",
                            f"tcp:{port}", "tcp:8080"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"端口转发失败: {e.stderr}")
            return False

    @staticmethod
    def check_camera_app_installed(device_id):
        """检查IP摄像头应用是否安装"""
        try:
            result = subprocess.run(
                ["adb", "-s", device_id, "shell", "pm", "list", "packages", "com.pas.webcam"],
                stdout=subprocess.PIPE, text=True
            )
            return "com.pas.webcam" in result.stdout
        except subprocess.CalledProcessError as e:
            print(f"检测应用失败: {e.stderr}")
            return False


# -------------------- 核心检测类 --------------------
class EmotionDetector:
    def __init__(self):
        # 下载并初始化模型
        self.download_models()

        # 加载YuNet人脸检测器
        self.face_detector = cv2.FaceDetectorYN_create(
            MODEL_PATH,
            "",
            (320, 320),
            0.9,  # 置信度阈值
            0.3,  # NMS阈值
            5000  # 最大检测数
        )

        # 加载DeepFace情绪识别模型
        self.emotion_net = cv2.dnn.readNetFromONNX(EMOTION_MODEL_PATH)
        self.emotion_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.emotion_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # 用于时间平滑的队列
        self.emotion_history = deque(maxlen=10)

    def download_models(self):
        """下载模型（如果不存在）"""
        # 下载人脸检测模型
        if not os.path.exists(MODEL_PATH):
            try:
                print("正在下载YuNet人脸检测模型...")
                urllib.request.urlretrieve(YUNET_MODEL_URL, MODEL_PATH)
                print("人脸检测模型下载完成")
            except Exception as e:
                print(f"人脸检测模型下载失败: {str(e)}")
                messagebox.showerror("错误", f"人脸检测模型下载失败: {str(e)}")

        # 下载情绪识别模型
        if not os.path.exists(EMOTION_MODEL_PATH):
            try:
                print("正在下载DeepFace情绪识别模型...")
                urllib.request.urlretrieve(DEEPFACE_MODEL_URL, EMOTION_MODEL_PATH)
                print("情绪识别模型下载完成")
            except Exception as e:
                print(f"情绪识别模型下载失败: {str(e)}")
                messagebox.showerror("错误", f"情绪识别模型下载失败: {str(e)}")

    def detect_faces(self, img, min_confidence=0.5):
        """使用YuNet检测人脸并返回坐标"""
        # 设置输入尺寸
        h, w = img.shape[:2]
        self.face_detector.setInputSize((w, h))

        # 检测人脸
        _, faces = self.face_detector.detect(img)
        if faces is None:
            return []

        # 提取人脸框
        detected_faces = []
        for face in faces:
            confidence = face[-1]
            if confidence < min_confidence:
                continue

            # 提取坐标 (x1, y1, w, h)
            x1 = int(face[0])
            y1 = int(face[1])
            x2 = int(face[0] + face[2])
            y2 = int(face[1] + face[3])

            # 确保边界框在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            width = x2 - x1
            height = y2 - y1

            detected_faces.append((x1, y1, width, height))

        return detected_faces

    def preprocess_face(self, face_img):
        """预处理人脸图像"""
        # 转换为灰度图
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # 直方图均衡化
        gray = cv2.equalizeHist(gray)

        # 调整大小为模型输入尺寸
        resized = cv2.resize(gray, (64, 64))

        # 归一化
        normalized = resized.astype(np.float32) / 255.0

        # 转换为3通道 (64, 64, 3)
        normalized = np.stack([normalized] * 3, axis=-1)

        return normalized

    def analyze_emotion(self, face_roi):
        """分析情绪（返回所有情绪概率）"""
        # 预处理
        preprocessed = self.preprocess_face(face_roi)

        # 转换为blob
        blob = cv2.dnn.blobFromImage(preprocessed, 1.0, (64, 64), (0, 0, 0), swapRB=False)

        # 预测
        self.emotion_net.setInput(blob)
        preds = self.emotion_net.forward()[0]
        probabilities = [float(p) for p in preds]

        # 时间平滑处理
        self.emotion_history.append(probabilities)
        smoothed_probs = np.mean(self.emotion_history, axis=0) if self.emotion_history else probabilities

        # 返回情绪字典
        return {
            emotion: prob
            for emotion, prob in zip(EMOTIONS, smoothed_probs)
        }

    def annotate_image(self, img, faces, emotions_list):
        """标注图像（显示所有情绪置信度）"""
        # 创建一个副本图像用于标注
        annotated = img.copy()

        # 1. 绘制人脸框
        for (x, y, w, h) in faces:
            # 绘制绿色方框标记人脸
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 使用PIL绘制中文文本
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(annotated_rgb)
        draw = ImageDraw.Draw(pil_img)

        try:
            # 尝试使用系统字体
            font = ImageFont.truetype("simhei.ttf", 16, encoding="utf-8")
            title_font = ImageFont.truetype("simhei.ttf", 18, encoding="utf-8")
        except:
            try:
                font = ImageFont.truetype("msyh.ttf", 16, encoding="utf-8")
                title_font = ImageFont.truetype("msyh.ttf", 18, encoding="utf-8")
            except:
                font = ImageFont.load_default()
                title_font = ImageFont.load_default()

        for i, ((x, y, w, h), emotions) in enumerate(zip(faces, emotions_list)):
            if not emotions:  # 如果情绪字典为空，跳过
                continue

            # 在图像右侧显示情绪热力图
            chart_x = x + w + 20
            chart_y = y
            chart_width = 150
            chart_height = 100

            # 绘制热力图背景
            draw.rectangle([(chart_x, chart_y), (chart_x + chart_width, chart_y + chart_height)],
                           fill=(30, 30, 30))

            # 绘制热力图标题
            draw.text((chart_x + 5, chart_y + 5), "情绪分析:", font=title_font, fill=(255, 255, 0))

            # 绘制情绪条
            bar_height = 12
            bar_spacing = 2
            y_offset = chart_y + 30

            # 找出置信度最高的情绪
            main_emotion, max_prob = max(emotions.items(), key=lambda x: x[1])

            for j, (emotion, prob) in enumerate(emotions.items()):
                # 计算条的长度
                bar_length = int(prob * 100)

                # 设置颜色（主要情绪用绿色）
                color = (0, 255, 0) if emotion == main_emotion else (200, 200, 200)

                # 绘制条形图
                bar_end = chart_x + 10 + bar_length
                draw.rectangle([(chart_x + 10, y_offset), (bar_end, y_offset + bar_height)], fill=color)

                # 绘制文本
                text = f"{emotion}: {prob:.1%}"
                draw.text((bar_end + 5, y_offset), text, font=font, fill=color)

                y_offset += bar_height + bar_spacing

            # 在方框上方显示主要情绪
            text = f"{main_emotion} ({max_prob:.1%})"
            text_width = draw.textlength(text, font=title_font)
            text_x = x + (w - text_width) / 2
            text_y = y - 30

            # 绘制背景矩形增强可读性
            draw.rectangle(
                [(text_x - 5, text_y - 5), (text_x + text_width + 5, text_y + 25)],
                fill=(0, 0, 0)
            )

            # 绘制文本
            draw.text((text_x, text_y), text, font=title_font, fill=(0, 255, 0))

        # 转换回OpenCV格式
        annotated = np.array(pil_img)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        return annotated


# -------------------- GUI界面 --------------------
class EmotionApp:

    def __init__(self, root):
        self.root = root
        self.detector = EmotionDetector()
        self.setup_ui()
        self.is_running = False
        self.cap = None
        self.last_update_time = 0
        self.adb_helper = ADBHelper()

        # 检查ADB是否安装
        if not self.adb_helper.check_adb_installed():
            messagebox.showwarning("警告", "未检测到ADB工具，请先安装Android SDK Platform-Tools")

        # 自动刷新设备列表
        self.refresh_devices()

    def optimize_stream(self):
        """优化视频流参数"""
        if self.source_var.get() == "安卓手机摄像头":
            # 设置较低分辨率
            subprocess.run([
                "adb", "-s", self.device_var.get(),
                "shell", "am", "broadcast", "-a", "com.pas.webcam.SET_FLAT",
                "--es", "config", "width=640;height=480;fps=15"
            ])

            # 设置OpenCV缓冲大小
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.cap.set(cv2.CAP_PROP_FPS, 15)

    def setup_ui(self):
        self.root.title("高精度情绪识别系统 - 基于YuNet和DeepFace模型")

        # 图片显示
        self.img_label = tk.Label(self.root)
        self.img_label.pack(pady=10)

        # 按钮区
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)

        tk.Button(btn_frame, text="上传图片", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="分析图片", command=self.run_detection).pack(side=tk.LEFT, padx=5)
        self.start_btn = tk.Button(btn_frame, text="开始实时识别", command=self.toggle_realtime)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        # 刷新设备按钮
        tk.Button(btn_frame, text="刷新设备", command=self.refresh_devices).pack(side=tk.LEFT, padx=5)

        # 设备选择
        device_frame = tk.Frame(self.root)
        device_frame.pack(pady=5)
        tk.Label(device_frame, text="选择设备:").pack(side=tk.LEFT)
        self.device_var = tk.StringVar()
        self.device_combobox = ttk.Combobox(device_frame, textvariable=self.device_var, width=30)
        self.device_combobox.pack(side=tk.LEFT, padx=5)

        # 视频源选择
        source_frame = tk.Frame(self.root)
        source_frame.pack(pady=5)
        tk.Label(source_frame, text="视频源:").pack(side=tk.LEFT)
        self.source_var = tk.StringVar(value="电脑摄像头")
        sources = ["电脑摄像头", "安卓手机摄像头"]
        for source in sources:
            tk.Radiobutton(source_frame, text=source, variable=self.source_var,
                           value=source).pack(side=tk.LEFT, padx=5)

        # 帧率控制
        fps_frame = tk.Frame(self.root)
        fps_frame.pack(pady=5)
        tk.Label(fps_frame, text="帧率限制:").pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="10")
        tk.Entry(fps_frame, textvariable=self.fps_var, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(fps_frame, text="FPS").pack(side=tk.LEFT)

        # 置信度阈值
        conf_frame = tk.Frame(self.root)
        conf_frame.pack(pady=5)
        tk.Label(conf_frame, text="人脸置信度阈值:").pack(side=tk.LEFT)
        self.conf_var = tk.StringVar(value="0.7")
        tk.Entry(conf_frame, textvariable=self.conf_var, width=5).pack(side=tk.LEFT, padx=5)

        # 操作指南
        guide_frame = tk.Frame(self.root)
        guide_frame.pack(pady=5)
        self.guide_label = tk.Label(guide_frame, text="", fg="blue")
        self.guide_label.pack()

        # 详细信息显示
        self.text = tk.Text(self.root, height=15, width=60)
        self.text.pack(pady=10)

    def refresh_devices(self):
        """刷新连接的设备列表"""
        devices = self.adb_helper.get_connected_devices()
        if devices:
            self.device_combobox['values'] = devices
            self.device_combobox.current(0)
            self.update_guide("")
        else:
            self.device_combobox['values'] = ["未检测到设备"]
            self.device_combobox.current(0)
            self.update_guide("未检测到设备，请检查:\n1. USB调试已开启\n2. 已授权电脑\n3. 使用原装数据线")

    def update_guide(self, message):
        """更新操作指南"""
        self.guide_label.config(text=message)

    def toggle_realtime(self):
        """切换实时识别状态"""
        if self.is_running:
            self.is_running = False
            self.start_btn.config(text="开始实时识别")
            if self.cap:
                self.cap.release()
                self.cap = None
            self.update_guide("实时识别已停止")
        else:
            # 根据选择的视频源初始化
            if self.source_var.get() == "电脑摄像头":
                # 使用电脑摄像头
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    messagebox.showerror("错误", "无法打开电脑摄像头")
                    return
            else:
                # 使用安卓手机摄像头
                device_id = self.device_var.get()
                if not device_id or device_id == "未检测到设备":
                    messagebox.showerror("错误", "请先连接安卓设备")
                    return

                # 检查是否安装了IP摄像头应用
                if not self.adb_helper.check_camera_app_installed(device_id):
                    messagebox.showerror("错误", "未检测到IP Webcam应用，请先安装")
                    return

                # 设置端口转发
                if not self.adb_helper.setup_forward(device_id):
                    messagebox.showerror("错误", "端口转发失败，请检查设备连接")
                    return

                # 提示用户手动操作
                self.update_guide("请在手机上:\n1. 打开IP Webcam应用\n2. 点击'启动服务器'")
                messagebox.showinfo("提示", "请手动启动IP Webcam服务后点击确定")

                # 尝试连接视频流
                self.cap = cv2.VideoCapture("http://localhost:8080/video")
                if not self.cap.isOpened():
                    messagebox.showerror("错误", "无法连接视频流，请确认:\n1. 服务已启动\n2. 端口转发成功")
                    return

            # 优化视频流参数
            self.optimize_stream()

            self.is_running = True
            self.start_btn.config(text="停止实时识别")
            self.text.delete(1.0, tk.END)
            self.text.insert(tk.END, "实时识别已启动...\n")
            self.update_guide("")

            # 启动实时检测线程
            threading.Thread(target=self.realtime_detection, daemon=True).start()

    def realtime_detection(self):
        """实时检测线程"""
        target_fps = float(self.fps_var.get())
        min_delay = 1.0 / target_fps
        conf_threshold = float(self.conf_var.get())

        while self.is_running:
            current_time = time.time()
            if current_time - self.last_update_time < min_delay:
                time.sleep(0.01)
                continue

            self.last_update_time = current_time

            ret, frame = self.cap.read()
            if not ret:
                if self.source_var.get() == "安卓手机摄像头":
                    self.text.insert(tk.END, "视频流中断，尝试重新连接...\n")
                    self.cap.release()
                    time.sleep(1)
                    self.cap = cv2.VideoCapture("http://localhost:8080/video")
                    continue
                else:
                    continue

            # 检测人脸
            try:
                faces = self.detector.detect_faces(frame, min_confidence=conf_threshold)
            except Exception as e:
                self.text.insert(tk.END, f"人脸检测错误: {str(e)}\n")
                continue

            # 分析每张脸的情绪
            emotions_list = []
            for (x, y, w, h) in faces:
                face_roi = frame[y:y + h, x:x + w]
                try:
                    emotions = self.detector.analyze_emotion(face_roi)
                    emotions_list.append(emotions)
                except Exception as e:
                    self.text.insert(tk.END, f"情绪分析错误: {str(e)}\n")
                    emotions_list.append({})  # 添加空字典保持列表长度一致

            # 显示标注后的图片
            try:
                annotated = self.detector.annotate_image(frame, faces, emotions_list)
                img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil.thumbnail((600, 600))  # 增大显示尺寸
                img_tk = ImageTk.PhotoImage(img_pil)
            except Exception as e:
                self.text.insert(tk.END, f"图像标注错误: {str(e)}\n")
                # 显示原始帧作为后备
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil.thumbnail((600, 600))
                img_tk = ImageTk.PhotoImage(img_pil)

            # 在主线程更新UI
            self.root.after(0, self.update_ui, img_tk, faces, emotions_list)

    def update_ui(self, img_tk, faces, emotions_list):
        """更新UI显示"""
        if not self.is_running:
            return

        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk

        # 更新文本框
        self.text.delete(1.0, tk.END)
        if len(faces) == 0:
            self.text.insert(tk.END, "未检测到人脸\n")
        else:
            for i, emotions in enumerate(emotions_list):
                if emotions:  # 确保情绪字典不为空
                    self.text.insert(tk.END, f"人脸 {i + 1} 情绪分析:\n")

                    # 找出最高置信度的情绪
                    main_emotion, max_prob = max(emotions.items(), key=lambda x: x[1])
                    self.text.insert(tk.END, f"  主要情绪: {main_emotion} ({max_prob:.2%})\n")

                    # 显示所有情绪概率
                    self.text.insert(tk.END, "  详细概率:\n")
                    for emotion, prob in emotions.items():
                        self.text.insert(tk.END, f"    {emotion}: {prob:.2%}\n")

                    self.text.insert(tk.END, "\n")
                else:
                    self.text.insert(tk.END, f"人脸 {i + 1} 情绪分析失败\n\n")

    def load_image(self):
        """加载图片"""
        if self.is_running:
            messagebox.showwarning("警告", "请先停止实时识别")
            return

        path = filedialog.askopenfilename(filetypes=[("图片", "*.jpg *.jpeg *.png")])
        if path:
            self.image = cv2.imread(path)
            if self.image is None:
                messagebox.showerror("错误", "无法加载图像文件")
                return
            self.show_thumbnail(path)
            self.text.delete(1.0, tk.END)

    def show_thumbnail(self, path):
        """显示缩略图"""
        img = Image.open(path)
        img.thumbnail((600, 600))  # 增大显示尺寸
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk

    def run_detection(self):
        """执行情绪分析"""
        if self.is_running:
            messagebox.showwarning("警告", "请先停止实时识别")
            return

        if not hasattr(self, 'image') or self.image is None:
            messagebox.showerror("错误", "请先上传图片")
            return

        try:
            conf_threshold = float(self.conf_var.get())
            faces = self.detector.detect_faces(self.image, min_confidence=conf_threshold)
        except Exception as e:
            messagebox.showerror("错误", f"人脸检测失败: {str(e)}")
            return

        if len(faces) == 0:
            messagebox.showinfo("提示", "未检测到人脸")
            return

        # 分析每张脸的情绪
        emotions_list = []
        for (x, y, w, h) in faces:
            face_roi = self.image[y:y + h, x:x + w]
            try:
                emotions = self.detector.analyze_emotion(face_roi)
                emotions_list.append(emotions)

                # 在文本框中显示详细结果
                self.text.insert(tk.END, f"人脸 {len(emotions_list)} 情绪分析:\n")

                # 找出最高置信度的情绪
                main_emotion, max_prob = max(emotions.items(), key=lambda x: x[1])
                self.text.insert(tk.END, f"  主要情绪: {main_emotion} ({max_prob:.2%})\n")

                # 显示所有情绪概率
                self.text.insert(tk.END, "  详细概率:\n")
                for emotion, prob in emotions.items():
                    self.text.insert(tk.END, f"    {emotion}: {prob:.2%}\n")

                self.text.insert(tk.END, "\n")
            except Exception as e:
                self.text.insert(tk.END, f"情绪分析失败: {str(e)}\n")
                emotions_list.append({})  # 添加空字典保持列表长度一致
                continue

        # 显示标注后的图片
        try:
            annotated = self.detector.annotate_image(self.image, faces, emotions_list)
            cv2.imshow("情绪分析结果", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            messagebox.showerror("错误", f"图像标注失败: {str(e)}")


# -------------------- 主程序 --------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()