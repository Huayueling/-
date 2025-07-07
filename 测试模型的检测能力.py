from ultralytics import YOLO
import cv2
import os

if __name__ == '__main__':
    # 1. 模型路径
    model_path = r"F:\pycahrm\untitled\write\dataset\model\tianzi_grid_model3\weights\best.pt"
    # 验证模型是否存在
    if not os.path.exists(model_path):
        model_path = model_path.replace("\\", "/")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在，请检查路径：{model_path}")

    # 加载模型
    model = YOLO(model_path)
    print(f"成功加载模型：{model_path}")

    # 2. 测试图片路径
    test_img_path = r"H:\书写评估\书法作业示例.jpg"
    if not os.path.exists(test_img_path):
        raise FileNotFoundError(f"测试图片不存在：{test_img_path}")

    # 3. 预测（只保留有字田字格，类别 0）
    results = model.predict(
        source=test_img_path,
        conf=0.3,  # 置信度阈值
        iou=0.45,
        device='0' if model.device.type == 'cuda' else 'cpu',  # 优先 GPU
        show=False
    )

    # 4. 绘制检测结果（只标有字田字格）
    for result in results:
        img = result.orig_img  # 原始图片
        for box in result.boxes:
            if box.cls == 0:  # 类别 0为有字田字格         这里很奇怪，我在标注数据的时候是把0标记的空白格子，1是有字的格子，但是这里刚好颠倒了
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 绘制蓝色框
                cv2.rectangle(img, (x1, y1), (x2, y2), (255,0, 0), 2)
                # 显示置信度
                conf = float(box.conf[0])
                cv2.putText(
                    img, f"conf: {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                )

        # 调整图像大小
        scale_factor = 0.3  # 缩放因子，范围 (0, 1) 表示缩小，大于 1 表示放大
        height, width = img.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        resized_img = cv2.resize(img, (new_width, new_height))

        # 直接显示调整大小后的结果
        cv2.imshow("检测结果", resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()