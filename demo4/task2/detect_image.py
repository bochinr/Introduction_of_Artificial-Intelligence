import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import os


class ObjectDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 目标检测器")
        self.root.geometry("1000x700")

        # 加载模型（首次运行会自动下载）
        self.model = YOLO('yolov8n.pt')

        # 创建界面组件
        self.create_widgets()

    def create_widgets(self):
        # 控制面板
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # 上传按钮
        self.btn_upload = tk.Button(
            control_frame,
            text="上传图片",
            command=self.upload_image,
            font=('Arial', 12),
            width=15
        )
        self.btn_upload.pack(side=tk.LEFT, padx=5)

        # 置信度滑块
        tk.Label(control_frame, text="置信度阈值:").pack(side=tk.LEFT, padx=5)
        self.conf_slider = tk.Scale(
            control_frame,
            from_=0.1,
            to=0.9,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.update_conf
        )
        self.conf_slider.set(0.5)
        self.conf_slider.pack(side=tk.LEFT, padx=5)

        # 图片显示区域
        image_frame = tk.Frame(self.root)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 原始图片区域
        original_frame = tk.LabelFrame(image_frame, text="原始图片", padx=5, pady=5)
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.label_original = tk.Label(original_frame, bg='#f0f0f0')
        self.label_original.pack(fill=tk.BOTH, expand=True)

        # 检测结果区域
        result_frame = tk.LabelFrame(image_frame, text="检测结果（带方框标注）", padx=5, pady=5)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.label_result = tk.Label(result_frame, bg='#f0f0f0')
        self.label_result.pack(fill=tk.BOTH, expand=True)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("准备就绪 - 请上传图片")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=('Arial', 10)
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png")]
        )

        if file_path:
            self.status_var.set("正在处理图片...")
            self.root.update()  # 立即更新界面

            try:
                # 读取原始图片
                orig_img = cv2.imread(file_path)
                orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

                # 显示原始图片
                self.show_image(orig_img_rgb, self.label_original, "原始图片")

                # 执行目标检测
                results = self.model(orig_img, conf=self.conf_slider.get())

                # 获取带标注的图片（自动包含方框、类别和置信度）
                annotated_img = results[0].plot()  # 这个方法会自动绘制方框
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

                # 显示检测结果
                self.show_image(annotated_img_rgb, self.label_result, "检测结果")

                # 保存结果
                save_path = os.path.splitext(file_path)[0] + "_detected.jpg"
                cv2.imwrite(save_path, annotated_img)

                # 显示检测到的目标信息
                detected_objects = []
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = result.names[class_id]
                        conf = float(box.conf)
                        detected_objects.append(f"{class_name} ({conf:.2f})")

                self.status_var.set(
                    f"检测完成！发现 {len(detected_objects)} 个目标: {', '.join(detected_objects)} | "
                    f"结果已保存到: {save_path}"
                )

            except Exception as e:
                messagebox.showerror("错误", f"检测失败: {str(e)}")
                self.status_var.set("检测出错")

    def show_image(self, cv_img, label_widget, title):
        # 将OpenCV图像转换为PIL格式
        img_pil = Image.fromarray(cv_img)

        # 计算缩放比例
        max_width = label_widget.winfo_width() - 20
        max_height = label_widget.winfo_height() - 20

        if max_width <= 0 or max_height <= 0:
            max_width, max_height = 400, 400

        # 保持宽高比缩放
        img_width, img_height = img_pil.size
        ratio = min(max_width / img_width, max_height / img_height)
        new_size = (int(img_width * ratio), int(img_height * ratio))
        img_resized = img_pil.resize(new_size, Image.LANCZOS)

        # 显示图片
        img_tk = ImageTk.PhotoImage(img_resized)
        label_widget.config(image=img_tk, text=title, compound=tk.TOP)
        label_widget.image = img_tk

    def update_conf(self, val):
        self.status_var.set(f"当前置信度阈值: {float(val):.2f} (重新上传图片生效)")


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectorApp(root)
    root.mainloop()