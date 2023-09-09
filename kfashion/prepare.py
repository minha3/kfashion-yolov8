import os
import imghdr
import ujson
import pickle
import yaml
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

FROM_JSON_NOT_FOUND_IMAGE_DATA = 0
FROM_JSON_NOT_FOUND_BBOX_DATA = 0
FROM_JSON_NOT_FOUND_DATASET_DETAIL = 0
FROM_JSON_NOT_FOUND_DATASET_COORDS = 0
FROM_JSON_SUCCEEDED = 0


LABEL_TRANSLATOR = {'아우터': 'outer', '상의': 'top', '하의': 'bottom', '원피스': 'dress'}


def prepare(source, dst):
    """

    :param source: root directory path where the image and label files are located
    :param dst: directory path to store results converted to yolo format
    :return:
    {`image_file_name`: {
        "image_path": {
            "root": string,
            "file": string
        },
        "label_path": {
            "root": string,
            "file" :string
        },
        "file_name": string,
        "width": int,
        "height": int,
        "bboxes": [{
            "label": string,
            "x1": int,
            "y1": int,
            "width" int,
            "height": int
        ]
    }
    """
    image_files = defaultdict(dict)

    with ThreadPoolExecutor() as executor:
        for root, dirs, files in os.walk(source):
            for file in files:
                executor.submit(check_file, image_files, root, file)

    print('FROM_JSON_NOT_FOUND_IMAGE_DATA', FROM_JSON_NOT_FOUND_IMAGE_DATA)
    print('FROM_JSON_NOT_FOUND_BBOX_DATA', FROM_JSON_NOT_FOUND_BBOX_DATA)
    print('FROM_JSON_SUCCEEDED', FROM_JSON_SUCCEEDED)

    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, 'kfashion.pickle'), 'wb') as f:
        pickle.dump(image_files, f)

    export_to_yolo(image_files, dst)


def check_file(image_files, root, file):
    path = os.path.join(root, file)
    if os.path.isfile(path):
        if imghdr.what(path) is not None:
            image_files[file]['image_path'] = {'root': root, 'file': file}
            image_files[file.strip("'")]['image_path'] = {'root': root, 'file': file}
        elif file.endswith('.json') and (data := parse_json_file(path)):
            image_files[data['file_name']]['label_path'] = {'root': root, 'file': file}
            image_files[data['file_name']].update(data)


def parse_json_file(path):
    global FROM_JSON_NOT_FOUND_IMAGE_DATA, FROM_JSON_NOT_FOUND_BBOX_DATA, FROM_JSON_SUCCEEDED
    with open(path, 'r') as f:
        data = ujson.load(f)
    result = {
        'file_name': '',
        'width': 0,
        'height': 0,
        'bboxes': []
    }

    if '이미지 정보' in data:
        if '이미지 파일명' in data['이미지 정보']:
            result['file_name'] = data['이미지 정보']['이미지 파일명']
        if '이미지 너비' in data['이미지 정보']:
            result['width'] = data['이미지 정보']['이미지 너비']
        if '이미지 높이' in data['이미지 정보']:
            result['height'] = data['이미지 정보']['이미지 높이']

    if not result['file_name'] or not result['width'] or not result['height']:
        FROM_JSON_NOT_FOUND_IMAGE_DATA += 1
        return

    if '데이터셋 정보' not in data:
        FROM_JSON_NOT_FOUND_BBOX_DATA += 1
        return
    if '데이터셋 상세설명' not in data['데이터셋 정보']:
        FROM_JSON_NOT_FOUND_BBOX_DATA += 1
        return
    if '렉트좌표' not in data['데이터셋 정보']['데이터셋 상세설명']:
        FROM_JSON_NOT_FOUND_BBOX_DATA += 1
        return

    for label_ko, bboxes in data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표'].items():
        if (label_en := translate(label_ko)) and bboxes:
            for bbox in bboxes:
                if bbox:
                    result['bboxes'].append({
                        'label': label_en,
                        'x1': bbox['X좌표'],
                        'y1': bbox['Y좌표'],
                        'width': bbox['가로'],
                        'height': bbox['세로']
                    })
    FROM_JSON_SUCCEEDED += 1
    return result


def translate(word):
    return LABEL_TRANSLATOR.get(word)


def export_to_yolo(image_files, dst):
    valid_image_files = []
    labels = set()
    for image_file_name, image_info in image_files.items():
        # filter out image files that don't have bounding boxes
        if image_info.get('image_path') and image_info.get('bboxes'):
            valid = True
            # collect all labels of bounding box
            for bbox in image_info['bboxes']:
                # filter out image files when bounding box coordinates or size exceed the image dimensions
                if (
                    bbox['x1'] + bbox['width'] > image_info['width']
                    or bbox['y1'] + bbox['height'] > image_info['height']
                ):
                    valid = False
                    break
                labels.add(bbox['label'])

            if valid:
                valid_image_files.append(image_info)

    label_to_idx = {l: i for i, l in enumerate(sorted(labels))}

    if not valid_image_files:
        print('No valid image files were found')
        return
    if not label_to_idx:
        print('No labels were found')
        return

    # set metadata
    config = {'path': dst, 'train': 'images/train', 'val': 'images/val', 'test': None,
              'nc': len(label_to_idx), 'names': {v: k for k, v in label_to_idx.items()}}

    with open(os.path.join(dst, 'kfashion.yaml'), 'w') as f:
        yaml.safe_dump(config, f)

    # separate images for training and validation
    cnt_train = max(1, int(len(valid_image_files) * 0.7))
    file_range_by_usage = {'train': range(0, cnt_train), 'val': range(cnt_train, len(valid_image_files))}

    for usage in ['train', 'val']:
        bbox_dir = os.path.join(dst, 'labels', usage)
        img_dir = os.path.join(dst, 'images', usage)
        os.makedirs(bbox_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        for i in file_range_by_usage[usage]:
            file_info = valid_image_files[i]
            origin_img_path = os.path.join(file_info['image_path']['root'], file_info['image_path']['file'])
            target_img_path = os.path.join(img_dir, file_info['image_path']['file'])
            os.symlink(origin_img_path, target_img_path)

            bbox_path = os.path.join(bbox_dir, change_file_extension(file_info['image_path']['file'], 'txt'))
            with open(bbox_path, 'w') as f:
                for bbox in file_info['bboxes']:
                    rwidth = bbox['width'] / file_info['width']
                    rheight = bbox['height'] / file_info['height']
                    rx1 = bbox['x1'] / file_info['width']
                    ry1 = bbox['y1'] / file_info['height']

                    f.write(
                        f"{label_to_idx[bbox['label']]}"
                        f" {rx1 + rwidth/2}"
                        f" {ry1 + rheight/2}"
                        f" {rwidth}"
                        f" {rheight}"
                        f"\n"
                    )


def change_file_extension(file_path, new_extension):
    file_name, extension = os.path.splitext(file_path)
    return f'{file_name}.{new_extension}'
