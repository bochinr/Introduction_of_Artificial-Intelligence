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

# -------------------- 配置部分 --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion-ferplus-8.onnx")
EMOTIONS = ["中性", "快乐", "惊讶", "悲伤", "愤怒", "厌恶", "恐惧"]  # ONNX模型的输出顺序

# 人脸检测模型配置
FACE_PROTOTXT = os.path.join(BASE_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")


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
        # 下载并初始化DNN人脸检测器
        self.download_face_model()
        self.face_net = cv2.dnn.readNetFromCaffe(FACE_PROTOTXT, FACE_MODEL)

        # 加载ONNX情绪识别模型
        if not os.path.exists(MODEL_PATH):
            self.download_emotion_model()
        self.emotion_net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        self.emotion_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.emotion_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def download_face_model(self):
        """下载人脸检测模型（如果不存在）"""
        # 配置文件
        if not os.path.exists(FACE_PROTOTXT):
            try:
                print("正在下载人脸检测配置文件...")
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                    FACE_PROTOTXT
                )
                print("配置文件下载完成")
            except Exception as e:
                print(f"人脸检测配置文件下载失败: {str(e)}")
                messagebox.showerror("错误", f"人脸检测配置文件下载失败: {str(e)}")

        # 模型文件
        if not os.path.exists(FACE_MODEL):
            try:
                print("正在下载人脸检测模型...")
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                    FACE_MODEL
                )
                print("模型文件下载完成")
            except Exception as e:
                print(f"人脸检测模型下载失败: {str(e)}")
                messagebox.showerror("错误", f"人脸检测模型下载失败: {str(e)}")

    def download_emotion_model(self):
        """下载情绪识别模型"""
        try:
            print("正在下载情绪识别模型...")
            urllib.request.urlretrieve(
                "https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
                MODEL_PATH
            )
            print("情绪模型下载完成")
        except Exception as e:
            print(f"情绪模型下载失败: {str(e)}")
            messagebox.showerror("错误", f"情绪模型下载失败: {str(e)}")

    def detect_faces(self, img, min_confidence=0.5):
        """使用DNN检测人脸并返回坐标"""
        # 获取图像尺寸
        (h, w) = img.shape[:2]

        # 预处理图像：调整大小并归一化
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        # 通过神经网络进行检测
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(0, detections.shape[2]):
            # 提取置信度
            confidence = detections[0, 0, i, 2]

            # 过滤掉低置信度的检测结果
            if confidence > min_confidence:
                # 计算边界框坐标
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # 确保边界框在图像范围内
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # 计算宽度和高度
                width = endX - startX
                height = endY - startY

                # 添加到人脸列表
                faces.append((startX, startY, width, height))

        return faces

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
        self.emotion_net.setInput(blob)
        preds = self.emotion_net.forward()[0]
        probabilities = [float(p) for p in preds]  # 转换为Python float类型

        # 返回情绪字典
        return {
            emotion: prob
            for emotion, prob in zip(EMOTIONS, probabilities)
        }

    def annotate_image(self, img, faces, emotions_list):
        """标注图像（只显示最高置信度的情绪）"""
        # 创建一个副本图像用于标注
        annotated = img.copy()

        # 1. 绘制人脸框
        for (x, y, w, h) in faces:
            # 绘制绿色方框标记人脸
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 使用PIL绘制中文文本
        # 转换颜色空间
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(annotated_rgb)
        draw = ImageDraw.Draw(pil_img)

        try:
            # 尝试使用系统字体（Windows）
            font = ImageFont.truetype("simhei.ttf", 24, encoding="utf-8")  # 增大字体大小
        except:
            try:
                # 尝试使用其他常见中文字体
                font = ImageFont.truetype("msyh.ttf", 24, encoding="utf-8")
            except:
                # 回退到默认字体（可能不支持中文）
                font = ImageFont.load_default()

        for i, ((x, y, w, h), emotions) in enumerate(zip(faces, emotions_list)):
            if not emotions:  # 如果情绪字典为空，跳过
                continue

            # 找出置信度最高的情绪
            main_emotion, max_prob = max(emotions.items(), key=lambda x: x[1])

            # 在方框下方显示主要情绪
            text = f"{main_emotion} ({max_prob:.1%})"

            # 计算文本宽度
            text_width = draw.textlength(text, font=font)

            # 计算文本位置（居中于人脸框下方）
            text_x = x + (w - text_width) / 2
            text_y = y + h + 5

            # 绘制背景矩形增强可读性
            bg_height = 30
            # 修复括号错误
            draw.rectangle(
                [(text_x - 5, text_y - 5), (text_x + text_width + 5, text_y + bg_height)],
                fill=(0, 0, 0)
            )

            # 绘制文本
            draw.text((text_x, text_y), text, font=font, fill=(0, 255, 0))

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
            # 设置较低分辨率（IP Webcam默认可能使用高清）
            subprocess.run([
                "adb", "-s", self.device_var.get(),
                "shell", "am", "broadcast", "-a", "com.pas.webcam.SET_FLAT",
                "--es", "config", "width=640;height=480;fps=15"
            ])

            # 设置OpenCV缓冲大小
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.cap.set(cv2.CAP_PROP_FPS, 15)

    def setup_ui(self):
        self.root.title("情绪识别系统 - 支持安卓手机摄像头 (DNN人脸检测)")

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
        self.fps_var = tk.StringVar(value="5")
        tk.Entry(fps_frame, textvariable=self.fps_var, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(fps_frame, text="FPS").pack(side=tk.LEFT)

        # 置信度阈值
        conf_frame = tk.Frame(self.root)
        conf_frame.pack(pady=5)
        tk.Label(conf_frame, text="人脸置信度阈值:").pack(side=tk.LEFT)
        self.conf_var = tk.StringVar(value="0.5")
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

            # 在图像上绘制人脸框（即使没有情绪分析）
            if len(faces) > 0:
                # 绘制绿色方框标记所有人脸
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
            else:
                emotions_list = []

            # 显示标注后的图片
            try:
                annotated = self.detector.annotate_image(frame, faces, emotions_list)
                img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil.thumbnail((400, 400))
                img_tk = ImageTk.PhotoImage(img_pil)
            except Exception as e:
                self.text.insert(tk.END, f"图像标注错误: {str(e)}\n")
                # 显示原始帧作为后备
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil.thumbnail((400, 400))
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
        img.thumbnail((400, 400))
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