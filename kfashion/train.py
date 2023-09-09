import os
from ultralytics import YOLO

PROJECT_NAME = 'kfashion'
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def train(data, epochs):
    weight_path = get_last_best_weight()
    model = YOLO(weight_path)
    model.add_callback('on_fit_epoch_end', export_best_to_onnx)
    model.train(data=data, epochs=epochs, batch=64, device="0,1")


def get_last_best_weight():
    model_root = os.path.join(DIR_PATH, '../runs/detect')

    if os.path.exists(model_root) and (train_dirs := os.listdir(model_root)):
        for train_dir in sorted(train_dirs, reverse=True):
            if train_dir.startswith('train'):
                weight_path = os.path.join(model_root, train_dir, 'weights', 'best.pt')
                if os.path.exists(weight_path):
                    return weight_path

    return 'yolov8n.pt'


def export_best_to_onnx(trainer):
    best_model = YOLO(str(trainer.best))
    best_model.export(format='onnx')
