import json
from typing import List

import glob
import os.path
import cv2
import numpy as np

import common


def identify_billboards(path: str) -> List[str]:
    pass


def write_data(file: str, data) -> None:
    with open(file, "w") as handle:
        json.dump(data, handle)


def read_data(file: str):
    with open(file) as handle:
        return json.load(handle)


def get_files_from_playing_for_benchmarks(
    source_dir: str,
    label_dir: str
) -> (List[str], List[str]):
    source_imgs = glob.glob(
        os.path.join(source_dir, os.path.join("train", "**", "*.jpg")),
        recursive=True
    )
    label_imgs = glob.glob(
        os.path.join(label_dir, os.path.join("train", "**", "*.png")),
        recursive=True
    )
    return source_imgs, label_imgs


def get_image_billboard_score(
    label_image_path: str
) -> int:
    label_image = cv2.imread(label_image_path)
    target_rgb = common.playing_for_benchmarks_billboard_colour
    target_bgr = target_rgb[::-1]
    billboard_pixel_count = np.sum(
        np.all(
            label_image == target_bgr,
            axis=2
        )
    )
    return int(billboard_pixel_count)


def identify_images_containing_billboards(
    source_images: List[str],
    label_images: List[str],
    batch_size: int = 250,
    save_file_path: str = None,
) -> (List[str], List[str]):
    """
    Given a list of source and label images,
    returns only those which contain a billboard.

    :param source_images: List of image paths
    :param label_images: List of image paths
    :param batch_size: Batch size
    :param save_file_path: Optional save file to persist progress
    :return:
    """
    data = {
        "version": 1,  # unused for now, allows backwards compatibility later if needed
        "schema": ["source_image", "label_image", "billboard_score"],
        "images": [],
    }
    if os.path.exists(save_file_path):
        data = read_data(save_file_path)
    known = {}
    for (source, label, score) in data["images"]:
        if score > 0:
            known[label] = True

    total_count = 0
    delta_count = 0

    for i, (source, label) in enumerate(zip(source_images, label_images)):
        if label not in known:
            score = get_image_billboard_score(label)
            data["images"] += [[
                source,
                label,
                score
            ]]
            if score > 0:
                total_count += 1
                delta_count += 1
        elif known[label] > 0:
            total_count += 1
            delta_count += 1
        if save_file_path is not None and i > 0 and i % batch_size == 0:
            print(
                f"[{i}/{len(source_images)} ({i / len(source_images) * 100:.1f}%)] Found {total_count}(+{delta_count} | {total_count / len(source_images) * 100:.1f}%) billboards so far.")
            write_data(save_file_path, data)
            delta_count = 0
    if save_file_path:
        write_data(save_file_path, data)

    # return source_billboard_images, label_billboard_images
    return zip(*[(entry[0], entry[1]) for entry in data["images"]])
