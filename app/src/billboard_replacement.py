# %%
import cv2
import numpy as np
import random
import scipy
# %%
from common import playing_for_benchmarks_billboard_colour

first = True


def display(image, scale_x=0.5, scale_y=0.5):
    global first
    shrunk = cv2.resize(image, (0, 0), fx=scale_x, fy=scale_y)
    cv2.imshow("Screen Location", shrunk)

    if cv2.waitKey(0 if first else 1000) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    first = False


def strip_colours(image, preserve=playing_for_benchmarks_billboard_colour):
    preserve_bgr = preserve[::-1]
    black = np.where(
        np.all(
            image == preserve_bgr,
            axis=-1
        )
    )
    image = 255 * np.ones(image.shape, dtype=np.uint8)
    image[black] = [0, 0, 0]
    return image


# noinspection PyDefaultArgument
def add_borders(image, border_color=[0, 0, 0], border_width=25):
    return cv2.copyMakeBorder(
        image,
        border_width,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_CONSTANT,
        value=border_color
    )


def remove_borders(image, border_width=25):
    return image[border_width:-border_width, border_width:-border_width]


def get_corner_points(contour, mode=1):
    if mode == 1:
        """
        https://docs.opencv.org/3.4/dc/d84/group__core__basic.html
        """
        contour = np.reshape(contour, (-1, 2))

        sorted_pts = np.zeros((4, 2), dtype=contour.dtype)
        s = np.sum(contour, axis=1)
        sorted_pts[0] = contour[np.argmin(s)]
        sorted_pts[2] = contour[np.argmax(s)]

        diff = np.diff(contour, axis=1)
        sorted_pts[1] = contour[np.argmin(diff)]
        sorted_pts[3] = contour[np.argmax(diff)]

        return sorted_pts
    if mode == 2:
        shape = np.shape(contour)
        contour = np.reshape(contour, (-1, 2))

        """
        https://stackoverflow.com/a/51075698
        """
        from functools import reduce
        import operator
        import math
        coords = contour # [(0, 1), (1, 0), (1, 1), (0, 0)]
        center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
        rtn = (sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360))
        rtn = np.reshape(rtn, shape)
        return rtn

def assert_points_not_too_close(points, min_dist = 5):
    dist = scipy.spatial.distance.cdist(points, points)
    dist = dist + np.identity(len(points)) * np.max(dist)
    if np.min(dist) < min_dist:
        raise ValueError("Points too close")

def add_ugly_contour_rect(image, contour, colour=(255, 255, 0)):
    """
    Adds a rectangle for the contour to the image.
    The rectangle is not skewed to match the contour.
    This function is only for experimentation and understanding the cv2.minAreaRect func
    :param image: Image
    :param contour: Contour points
    :param colour: Rect colour
    :return: Image with rectangle in it
    """
    rect = cv2.minAreaRect(contour)
    rect = cv2.boxPoints(rect)
    rect = np.int32(rect)
    rect = get_corner_points(rect)
    image = cv2.fillConvexPoly(image, rect, colour)
    return image


def reduce_polygon(points, target_count=4):
    modifier = 0.02
    allowed_iterations = 100000
    original = points
    while len(points) > target_count and allowed_iterations > 0:
        perimeter = cv2.arcLength(points, True)
        points = cv2.approxPolyDP(
            original,
            modifier * perimeter,
            True
        )
        modifier += 0.0001
        allowed_iterations -= 1

    return points


# noinspection PyShadowingNames
def get_billboard_contours(label_image, minimum_area=25):
    border_width = 25
    padded_image = add_borders(label_image, border_width=border_width)
    binary_image = strip_colours(padded_image)
    edged_image = cv2.Canny(binary_image, 30, 200)

    (contours, _) = cv2.findContours(
        edged_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Remove small contours
    contours = [
        contour
        for contour
        in contours
        if cv2.contourArea(contour) >= minimum_area
    ]

    # Sort highest area to lowest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Convert to 4-pointed polygon
    contours = [
        reduce_polygon(contour)
        for contour
        in contours
    ]

    # Ensure at least 4 points describing the contour (required for homography function later)
    # contours = [contour for contour in contours if len(contour) >= 4]

    # Remove artificial padding
    contours = np.subtract(contours, border_width)
    return contours


# noinspection PyShadowingNames
def get_contour_mask(image_shape, contour, mode=3):
    image = 255 * np.ones(image_shape, dtype=np.uint8)
    if mode == 1:
        # Not recommended since fill sometimes doesn't fill properly...
        cv2.drawContours(
            image=image,
            contours=[contour],
            contourIdx=-1,
            color=(0, 0, 0),
            thickness=cv2.FILLED
        )
    elif mode == 2:
        # Not recommended since contour sometimes doesn't match homography warping...
        image = cv2.fillConvexPoly(
            image,
            contour,
            (0,0,0)
        )
    elif mode == 3:
        decal = 255 * np.ones(image_shape, dtype=np.uint8)
        white_decal_black_background = get_homo_warped_decal(image_shape, decal, contour)
        black_decal_white_background = np.invert(white_decal_black_background)
        image = black_decal_white_background
    return image



# noinspection PyShadowingNames
def get_homo_warped_decal(destination_shape, decal, contour):
    contour = np.reshape(contour, (-1, 2))

    assert_points_not_too_close(contour)

    h_decal, w_decal = decal.shape[:2]
    h_background, w_background = destination_shape[:2]
    src = np.float32([
        [0, 0],
        [w_decal, 0],
        [w_decal, h_decal],
        [0, h_decal],
    ])

    h, mask, = cv2.findHomography(src, contour, cv2.RANSAC, 5.0)

    warped = cv2.warpPerspective(decal, h, (w_background, h_background))
    return warped


def match_decal_to_source_colours(decal_image, source_image, mode: str = "deepbillboardpaper"):
    import skimage.exposure
    import color_transfer
    import old.utils
    if mode == "skimage":
        return skimage.exposure.match_histograms(decal_image, source_image, multichannel=True)
    elif mode == "color_transfer":
        return color_transfer.color_transfer(source_image, decal_image)
    elif mode == "deepbillboardpaper":
        return old.utils.control_bound(decal_image)
    else:
        raise ValueError(f"Match mode {mode} not supported.")


def apply_image_to_contour(source_image, decal_image, contour):
    mask_image = get_contour_mask(source_image.shape, contour)

    # remove the old billboard decal
    source_image_no_billboard = cv2.bitwise_and(source_image, mask_image)

    # warp the decal to fit the billboard
    warped_image = get_homo_warped_decal(source_image.shape, decal_image, contour)

    # apply the decal to the billboard
    output_image = cv2.bitwise_or(source_image_no_billboard, warped_image)

    return output_image


def show_contour_outlines(image, contours, colour=[255, 255, 0]):
    return cv2.drawContours(
        image.copy(),
        contours,
        contourIdx=-1,
        color=colour,
        thickness=5
    )


def show_contour_points(image, contour):
    image = image.copy()
    colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
    for i, center in enumerate([x[0] for x in contour]):
        cv2.circle(image, tuple(center), 20, colors[i], -1)
        cv2.putText(
            image,
            str(i),
            (center[0]-7, center[1]+7),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,0)
        )
    return image


# noinspection PyShadowingNames
def apply_image_to_billboard(source_image, label_image, decal_image):
    """
    Applies a decal image to the source image.
    Determines the location based on the largest detected billboard in the labeled image.
    :param source_image: Source image
    :param label_image: Source image but labelled with flat colours
    :param decal_image: Image to be applied
    :return: Output image, contour used, intermediary steps
    """
    decal = match_decal_to_source_colours(decal_image, source_image)
    contour = get_billboard_contours(label_image)[0]  # Grab largest
    output_image = apply_image_to_contour(source_image, decal, contour)
    return output_image, contour


# def apply_images_to_contours(source_image, decal_images, contours):
#     return [
#         apply_image_to_contour(source_image, decal, contour)
#         for decal, contour
#         in zip(decal_images, contours)
#     ]


def apply_images_to_billboards(source_image, label_image, decal_images, single_image: bool = True):  # Union[List[Image], Image]
    contours = get_billboard_contours(label_image)
    contours = [get_corner_points(c) for c in contours]
    decals = random.choices(decal_images, k=len(contours))
    decals = [match_decal_to_source_colours(decal, source_image) for decal in decals]
    output_images = []
    failures = []
    for contour, decal in zip(contours, decals):
        try:
            image = apply_image_to_contour(source_image, decal, contour)
            output_images.append(image)
            if single_image:
                # Feed output image back in so that all billboards are replaced into a single image
                source_image = image
        except:
            failures.append(contour)
    return source_image if single_image else output_images, contours, failures
