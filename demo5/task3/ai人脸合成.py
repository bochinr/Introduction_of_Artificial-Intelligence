import cv2
import numpy as np
import dlib
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import threading


class FaceMorpherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ai人脸合成")
        self.root.geometry("1000x700")

        # 初始化dlib模型
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        except Exception as e:
            messagebox.showerror("错误", f"初始化失败: {str(e)}")
            self.root.destroy()
            return

        # 状态变量
        self.img1 = None
        self.img2 = None
        self.img1_path = ""
        self.img2_path = ""
        self.alpha = 0.5
        self.processing = False

        # 创建GUI
        self.create_widgets()

    def create_widgets(self):
        # 主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 控制面板
        control_frame = tk.Frame(main_frame, bd=2, relief=tk.RIDGE)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # 图片上传按钮
        tk.Label(control_frame, text="上传图片", font=('Arial', 12, 'bold')).pack(pady=5)

        self.btn_img1 = tk.Button(control_frame, text="上传图片1", command=lambda: self.upload_image(1))
        self.btn_img1.pack(fill=tk.X, pady=5)

        self.btn_img2 = tk.Button(control_frame, text="上传图片2", command=lambda: self.upload_image(2))
        self.btn_img2.pack(fill=tk.X, pady=5)

        # 融合系数滑块
        tk.Label(control_frame, text="融合系数", font=('Arial', 10)).pack(pady=(10, 0))
        self.alpha_slider = tk.Scale(control_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL,
                                     command=self.update_alpha)
        self.alpha_slider.set(0.5)
        self.alpha_slider.pack(fill=tk.X, pady=5)

        # 开始合成按钮
        self.btn_start = tk.Button(control_frame, text="开始合成", state=tk.DISABLED,
                                   command=self.start_morphing)
        self.btn_start.pack(fill=tk.X, pady=10)

        # 进度条
        self.progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)

        # 图片显示区域
        display_frame = tk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 原始图片显示
        self.img1_label = tk.Label(display_frame, text="图片1未上传", bd=1, relief=tk.SUNKEN)
        self.img1_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.img2_label = tk.Label(display_frame, text="图片2未上传", bd=1, relief=tk.SUNKEN)
        self.img2_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 结果显示区域
        self.result_label = tk.Label(display_frame, text="等待合成...", bd=1, relief=tk.SUNKEN)
        self.result_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def upload_image(self, img_num):
        file_path = filedialog.askopenfilename(
            title=f"选择图片{img_num}",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            try:
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError("无法读取图片")

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)

                # 调整图片大小以适应显示区域
                max_size = (400, 300)
                img_pil.thumbnail(max_size, Image.LANCZOS)

                img_tk = ImageTk.PhotoImage(img_pil)

                if img_num == 1:
                    self.img1 = img
                    self.img1_path = file_path
                    self.img1_label.config(image=img_tk, text="")
                    self.img1_label.image = img_tk
                else:
                    self.img2 = img
                    self.img2_path = file_path
                    self.img2_label.config(image=img_tk, text="")
                    self.img2_label.image = img_tk

                # 检查是否可以启用开始按钮
                if self.img1 is not None and self.img2 is not None:
                    self.btn_start.config(state=tk.NORMAL)

            except Exception as e:
                messagebox.showerror("错误", f"加载图片失败: {str(e)}")

    def update_alpha(self, val):
        self.alpha = float(val)

    def start_morphing(self):
        if self.processing:
            return

        self.processing = True
        self.btn_start.config(state=tk.DISABLED)
        self.progress["value"] = 0

        # 在新线程中执行耗时操作
        threading.Thread(target=self.process_morphing, daemon=True).start()

    def process_morphing(self):
        try:
            # 调整图像大小一致
            h, w = min(self.img1.shape[0], self.img2.shape[0]), min(self.img1.shape[1], self.img2.shape[1])
            img1 = cv2.resize(self.img1, (w, h))
            img2 = cv2.resize(self.img2, (w, h))

            # 获取人脸关键点
            points1 = self.detect_face_landmarks(img1)
            points2 = self.detect_face_landmarks(img2)

            if points1 is None or points2 is None:
                raise ValueError("未能检测到人脸，请确保图像中包含清晰的人脸")

            # 获取三角剖分
            triangles = self.get_triangles(points1, (w, h))

            # 显示三角剖分结果
            self.show_triangles(img1, points1, triangles, "图片1三角剖分")
            self.show_triangles(img2, points2, triangles, "图片2三角剖分")

            # 合成人脸
            morphed_img = self.morph_images(img1, img2, points1, points2, triangles, self.alpha)

            # 显示结果
            self.show_result(morphed_img)

        except Exception as e:
            messagebox.showerror("错误", f"合成失败: {str(e)}")
        finally:
            self.processing = False
            self.progress["value"] = 100
            self.root.after(100, lambda: self.btn_start.config(state=tk.NORMAL))

    def detect_face_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            return None

        landmarks = self.predictor(gray, faces[0])
        return [(point.x, point.y) for point in landmarks.parts()]

    def get_triangles(self, points, image_size):
        w, h = image_size
        rect = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)

        # 添加图像边界点确保覆盖整个图像
        boundary_points = self.generate_boundary_points(image_size)
        all_points = points + boundary_points

        # 插入所有点
        for p in all_points:
            subdiv.insert(p)

        # 获取三角形列表
        triangle_list = subdiv.getTriangleList()

        # 转换为点索引形式
        triangles = []
        for t in triangle_list:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            indices = []
            for p in [pt1, pt2, pt3]:
                for i, point in enumerate(all_points):
                    if abs(point[0] - p[0]) < 1.0 and abs(point[1] - p[1]) < 1.0:
                        indices.append(i)
                        break

            if len(indices) == 3:
                triangles.append((indices[0], indices[1], indices[2]))

        return triangles

    def generate_boundary_points(self, image_size, num_points=16):
        w, h = image_size
        points = []

        # 上边界
        for x in np.linspace(0, w - 1, num=num_points // 4 + 2)[1:-1]:
            points.append((int(x), 0))

        # 右边界
        for y in np.linspace(0, h - 1, num=num_points // 4 + 2)[1:-1]:
            points.append((w - 1, int(y)))

        # 下边界
        for x in np.linspace(0, w - 1, num=num_points // 4 + 2)[1:-1]:
            points.append((int(x), h - 1))

        # 左边界
        for y in np.linspace(0, h - 1, num=num_points // 4 + 2)[1:-1]:
            points.append((0, int(y)))

        # 四个角点
        points.extend([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)])

        return points

    def show_triangles(self, img, points, triangles, title):
        # 绘制三角形
        vis_img = img.copy()

        # 获取所有点(包括边界点)
        boundary_points = self.generate_boundary_points((img.shape[1], img.shape[0]))
        all_points = points + boundary_points

        # 绘制三角形
        for tri in triangles:
            pt1 = all_points[tri[0]]
            pt2 = all_points[tri[1]]
            pt3 = all_points[tri[2]]

            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
            cv2.line(vis_img, pt2, pt3, (0, 255, 0), 1)
            cv2.line(vis_img, pt3, pt1, (0, 255, 0), 1)

        # 绘制关键点
        for i, (x, y) in enumerate(all_points):
            color = (255, 0, 0) if i < len(points) else (0, 0, 255)  # 人脸点蓝色，边界点红色
            cv2.circle(vis_img, (x, y), 2, color, -1)

        # 显示在新窗口中
        self.show_image_in_window(vis_img, title)

    def morph_images(self, img1, img2, points1, points2, triangles, alpha):
        h, w = img1.shape[:2]

        # 生成边界点
        boundary_points = self.generate_boundary_points((w, h))
        all_points1 = points1 + boundary_points
        all_points2 = points2 + boundary_points

        # 计算融合后的关键点
        morphed_points = self.compute_morph_points(all_points1, all_points2, alpha)

        # 创建空白图像
        morphed_img = np.zeros((h, w, 3), dtype=np.float32)

        total_triangles = len(triangles)

        for i, tri_indices in enumerate(triangles):
            # 更新进度
            progress = int((i + 1) / total_triangles * 100)
            self.root.after(10, lambda v=progress: self.progress.config(value=v))

            # 获取三个顶点
            tri1 = np.array([all_points1[i] for i in tri_indices], dtype=np.float32)
            tri2 = np.array([all_points2[i] for i in tri_indices], dtype=np.float32)
            morphed_tri = np.array([morphed_points[i] for i in tri_indices], dtype=np.float32)

            # 变形两个图像的三角形区域
            warped_tri1 = self.warp_triangle(img1, tri1, morphed_tri, (w, h))
            warped_tri2 = self.warp_triangle(img2, tri2, morphed_tri, (w, h))

            # 创建三角形mask
            mask = np.zeros((h, w, 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(morphed_tri), (1.0, 1.0, 1.0), 16, 0)

            # 混合两个变形后的三角形
            morphed_tri = (1.0 - alpha) * warped_tri1 + alpha * warped_tri2

            # 将混合后的三角形添加到结果图像
            morphed_img = morphed_img * (1 - mask) + morphed_tri * mask

        return np.uint8(morphed_img)

    def compute_morph_points(self, points1, points2, alpha):
        return [
            (int((1 - alpha) * x1 + alpha * x2), int((1 - alpha) * y1 + alpha * y2))
            for (x1, y1), (x2, y2) in zip(points1, points2)
        ]

    def warp_triangle(self, img, src_tri, dst_tri, size):
        warp_mat = cv2.getAffineTransform(src_tri, dst_tri)
        warped = cv2.warpAffine(img, warp_mat, size, None,
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        return warped

    def show_image_in_window(self, img, title):
        # 创建一个新窗口显示图片
        window = tk.Toplevel(self.root)
        window.title(title)

        # 调整图片大小以适应窗口
        img_pil = Image.fromarray(img)
        max_size = (800, 600)
        img_pil.thumbnail(max_size, Image.LANCZOS)

        img_tk = ImageTk.PhotoImage(img_pil)

        label = tk.Label(window, image=img_tk)
        label.image = img_tk
        label.pack()

        # 添加关闭按钮
        btn_close = tk.Button(window, text="关闭", command=window.destroy)
        btn_close.pack(pady=5)

    def show_result(self, img):
        # 准备显示的图片
        img_pil = Image.fromarray(img)
        max_size = (600, 500)
        img_pil.thumbnail(max_size, Image.LANCZOS)

        img_tk = ImageTk.PhotoImage(img_pil)

        # 更新结果显示区域
        self.result_label.config(image=img_tk, text="")
        self.result_label.image = img_tk

        # 保存结果按钮
        btn_save = tk.Button(self.root, text="保存结果", command=lambda: self.save_result(img))
        btn_save.pack(pady=5)

    def save_result(self, img):
        file_path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
        )

        if file_path:
            try:
                cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                messagebox.showinfo("成功", f"结果已保存到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceMorpherApp(root)
    root.mainloop()