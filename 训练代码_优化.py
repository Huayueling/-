from ultralytics import YOLO
import os
import torch

if __name__ == '__main__':

    print(f"使用设备: GPU ({torch.cuda.get_device_name(0)})")
    print(f"CUDA版本: {torch.version.cuda}")

    model_save_dir = r"F:\pycahrm\untitled\write\dataset\model"
    os.makedirs(model_save_dir, exist_ok=True)

    data_path = "F:/pycahrm/untitled/write/dataset/data.yaml"
    model_type = "yolov8s.pt"
    epochs = 300
    batch_size = 16
    imgsz = 640

    model = YOLO(model_type)

    print("开始GPU训练...")
    results = model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project=model_save_dir,
        name="tianzi_grid_model",
        lr0=0.001,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        device=0,
        amp=True,
        workers=4,
        patience=50,
        save_period=20,
        cache='disk'
    )

    metrics = model.val()
    print(f"\n训练完成！mAP@0.5: {metrics.box.map50:.4f}")