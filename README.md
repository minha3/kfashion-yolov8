# KFashion Object Detection with YOLOv8

This project utilizes the YOLOv8 architecture to perform object detection on the KFashion dataset.

## DataSet

- **KFashion DataSet**: This dataset is provided by AI HUB. For detailed information and downloads, please refer to [this link](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=51).

## Prerequisites

- **Python**: The project was tested on `Python 3.10.11`. You can download Python from its [official website](https://www.python.org/downloads/).
- **YOLOv8**: We use the YOLOv8 implementation by Ultralytics. For detailed implementation and further explanations, visit the [Ultralytics GitHub page](https://github.com/ultralytics/ultralytics).

## Installation

```bash
git clone https://github.com/minha3/kfashion-yolov8.git
cd kfashion-yolov8

pip install -r requirements.txt
```

## Pre-training Preparation

1. **KFashion DataSet Preparation**: Ensure that the downloaded KFashion dataset is placed in the `/opt/dataset/kfashion` directory.
2. **Generate Training YAML**: Execute the command below. It will create a `kfashion.yaml` file inside the `$HOME/dataset/kfashion` directory, which is required for training.

```bash
python main.py --prepare
```

## Training

1. **Model Training**: By executing the following command, the model will be trained using the `$HOME/dataset/kfashion/kfashion.yaml`.
2. **Output**: Once the training completes, the outputs will be stored in the `runs/detect` folder inside the project directory.

```bash
python main.py --train
```

## Results

Training and test results can be found in `kfashion-yolov8/runs/detect`.

## License
[MIT](LICENSE)
