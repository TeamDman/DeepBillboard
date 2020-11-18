import json
from typing import List, Tuple, Any, Callable, Dict

import glob
import os.path
import cv2
import numpy as np

import common


def write_data(file: str, data: Any) -> None:
    with open(file, "w") as handle:
        json.dump(data, handle)


def read_data(file: str) -> Any:
    with open(file) as handle:
        data = json.load(handle)
        return data


def get_billboards_from_data(
    data: Dict,
    score_pred: Callable[[int], bool] = lambda x: x > 0
) -> Tuple[List[str], List[str]]:
    billboard_source_images, \
    billboard_label_images = zip(*[
        (entry[0], entry[1])
        for entry
        in data["images"]
        if score_pred(entry[2])
    ])
    return billboard_source_images, billboard_label_images


def get_files_from_playing_for_benchmarks(
    source_dir: str,
    label_dir: str
) -> (List[str], List[str]):
    source_images = glob.glob(
        os.path.join(source_dir, os.path.join("train", "**", "*.jpg")),
        recursive=True
    )
    label_images = glob.glob(
        os.path.join(label_dir, os.path.join("train", "**", "*.png")),
        recursive=True
    )
    return source_images, label_images


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
    score_func: Callable[[str], int] = get_image_billboard_score,
    score_pred: Callable[[int], bool] = lambda x: x > 0,
    read_only: bool = False,
) -> (List[str], List[str]):
    """
    Given a list of source and label images,
    returns only those which contain a billboard.

    :param score_pred: Given a score, return true if it meets the criteria for a billboard
    :param score_func: Func given label file path, produce int representing billboard score. Higher score means better billboard candidate.
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
        print("Found existing data file.")
        data = read_data(save_file_path)
    else:
        print("Data file does not yet exist.")

    known = {}
    for (source, label, score) in data["images"]:
        known[label] = score

    total_count = 0
    delta_count = 0

    for i, (source, label) in enumerate(zip(source_images, label_images)):
        if label not in known:
            score = score_func(label)
            data["images"] += [[
                source,
                label,
                score
            ]]
            if score > 0:
                total_count += 1
                delta_count += 1
        elif score_pred(known[label]):
            total_count += 1
            delta_count += 1
        if batch_size > 0 and save_file_path is not None and i > 0 and i % batch_size == 0:
            print(
                f"[{i}/{len(source_images)} ({i / len(source_images) * 100:.1f}%)] Found {total_count}(+{delta_count} | {total_count / len(source_images) * 100:.1f}%) billboards so far.")
            if not read_only:
                write_data(save_file_path, data)
            delta_count = 0
    if save_file_path and not read_only:
        write_data(save_file_path, data)

    # return source_billboard_images, label_billboard_images
    billboard_source_images,\
    billboard_label_images = get_billboards_from_data(data, score_pred)

    print(f"Billboards present in {total_count}/{len(source_images)} ({total_count/len(source_images) * 100:.1f}%).")
    return billboard_source_images, billboard_label_images
