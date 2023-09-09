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

## Training

To run the `main.py` script, you should follow these steps:

1. **KFashion DataSet Preparation**: Ensure that the downloaded KFashion dataset is placed in the `/opt/dataset/kfashion` directory.

2. **Execution of main.py**: When you run the `main.py` script, a necessary YAML file for training will be generated in the `$HOME/dataset/kfashion` directory, named as `kfashion.yaml`.

3. **Model Training**: Use the `kfashion.yaml` file to start the training process.

4. **Output**: Once the training completes, the outputs will be stored in the `runs/detect` folder inside the project directory.

To train the model, use the following command:

```bash
python main.py
```

## Results

Training and test results can be found in `kfashion-yolov8/runs/detect`.

## License
[MIT](LICENSE)
