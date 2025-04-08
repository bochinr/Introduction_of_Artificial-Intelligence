import numpy as np
import cv2
import os
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 常量定义
MODEL_PATH = 'mnist_model.h5'
INPUT_SIZE = 28


def train_or_load_model():
    """加载模型：存在则加载，不存在则训练"""
    if os.path.exists(MODEL_PATH):
        print(f"检测到已有模型 {MODEL_PATH}，正在加载...")
        return models.load_model(MODEL_PATH)
    else:
        # 加载数据
        (train_images, train_labels), _ = mnist.load_data()

        # 数据预处理
        train_images = train_images.reshape((60000, INPUT_SIZE * INPUT_SIZE)).astype('float32') / 255
        train_labels = to_categorical(train_labels)

        # 构建模型
        model = models.Sequential([
            layers.Dense(512, activation='relu', input_shape=(INPUT_SIZE * INPUT_SIZE,)),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # 训练模型（简化版，仅用训练集）
        model.fit(train_images, train_labels, epochs=5, batch_size=128)
        model.save(MODEL_PATH)
        print(f"模型已保存至 {MODEL_PATH}")
        return model


def predict_digit(model, image_array):
    """预测手写数字"""
    processed_img = image_array.reshape(1, INPUT_SIZE * INPUT_SIZE).astype('float32') / 255
    return np.argmax(model.predict(processed_img))


def drawing_window():
    """手写绘图窗口"""
    canvas = np.zeros((280, 280), dtype=np.uint8)
    cv2.namedWindow('Draw Digit (Enter:识别 | ESC:退出)')

    def draw(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or flags == cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(canvas, (x, y), 12, 255, -1)

    cv2.setMouseCallback('Draw Digit (Enter:识别 | ESC:退出)', draw)

    while True:
        # 显示画板（添加操作提示）
        display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        cv2.putText(display, "Left Drag: Draw | Enter: Predict | ESC: Exit",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('Draw Digit (Enter:识别 | ESC:退出)', display)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # 回车识别
            resized = cv2.resize(canvas, (INPUT_SIZE, INPUT_SIZE))
            digit = predict_digit(model, resized)
            print(f"识别结果: {digit}")
            canvas.fill(0)
        elif key == 27:  # ESC退出
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 自动处理模型加载/训练
    model = train_or_load_model()

    # 直接进入绘图交互界面
    drawing_window()