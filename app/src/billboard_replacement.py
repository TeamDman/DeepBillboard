# %%
import cv2
import numpy as np

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
    black_indices = np.where((image != preserve).all(axis=2))
    white_indices = np.where((image == preserve).all(axis=2))
    rtn = image.copy()
    rtn[black_indices] = [0, 0, 0]
    rtn[white_indices] = [255, 255, 255]
    return rtn


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


# noinspection PyShadowingNames
def get_billboard_contours(label_image):
    border_width = 25
    padded_image = add_borders(label_image, border_width=border_width)
    binary_image = strip_colours(padded_image)
    edged_image = cv2.Canny(binary_image, 30, 200)

    (contours, _) = cv2.findContours(
        edged_image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort highest area to lowest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Convert contours to points
    contours = [
        cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        for contour
        in contours
    ]

    # Ensure at least 4 points describing the contour (required for homography function later)
    contours = [contour for contour in contours if len(contour) >= 4]

    # Remove artificial padding
    contours = np.subtract(contours, border_width)
    return contours


# noinspection PyShadowingNames
def get_contour_mask(image, contour):
    rtn = image.copy()
    cv2.drawContours(
        image=rtn,
        contours=[contour],
        contourIdx=-1,
        color=(255, 255, 255),
        thickness=cv2.FILLED
    )
    return rtn


def rotate(lst, x):
    return lst[-x:] + lst[:-x]


# noinspection PyShadowingNames
def get_homo_warped_decal(background, decal, contour):
    contour = [[x[0]] for x in contour]
    # contour = contour[
    contour = rotate(contour, 1)
    contour = np.float32(contour)

    h_decal, w_decal = decal.shape[:2]
    h_background, w_background = background.shape[:2]
    src = np.float32([
        [0, h_decal],
        [0, 0],
        [w_decal, 0],
        [w_decal, h_decal],
    ])
    dst = np.float32(contour)
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    flipped = np.flip(decal, axis=1)

    warped = cv2.warpPerspective(flipped, h, (w_background, h_background))
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
    blank_image = np.zeros(source_image.shape, dtype=np.uint8)
    mask_image = get_contour_mask(blank_image, contour)
    mask_inv_image = cv2.bitwise_not(mask_image)

    # remove the old billboard decal
    source_image_no_billboard = cv2.bitwise_and(source_image, mask_inv_image)

    # warp the decal to fit the billboard
    warped_image = get_homo_warped_decal(source_image, decal_image, contour)

    # apply the decal to the billboard
    output_image = cv2.bitwise_or(source_image_no_billboard, warped_image)

    return output_image


def show_contour_outline(image, contour):
    return cv2.drawContours(
        image.copy(),
        [contour],
        -1,
        [0, 255, 0],
        3
    )


def show_contour_points(image, contour):
    image = image.copy()
    colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
    for i, center in enumerate([x[0] for x in contour]):
        cv2.circle(image, tuple(center), 20, colors[i], -1)
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


def apply_images_to_billboards(source_image, label_image, decal_images):
    contours = get_billboard_contours(label_image)
    decals = np.random.choice(decal_images, len(contours))
    decals = [match_decal_to_source_colours(decal) for decal in decals]
    output_images = [
        apply_image_to_contour(source_image, decal, contour)
        for decal, contour
        in zip(decals, contours)
    ]
    return output_images, contours
