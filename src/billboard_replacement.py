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


# %%

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
def get_largest_billboard_contour(image):
    edged_image = cv2.Canny(image, 30, 200)

    (contours, _) = cv2.findContours(
        edged_image.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )[:10]

    best = (0, 0)

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        area = cv2.contourArea(contour)
        # print("Found contour with {} nodes and total area {}"
        #       .format(len(approx), area))
        # if len(approx) == 4:
        if area > best[0]:
            best = (area, approx)

    return best[1] if best[0] > 0 else None


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


# %%
# noinspection PyShadowingNames
def apply_image_to_billboard(source_image, labeled_source_image, decal_image):
    # pad image so billboards touching the edge can be detected properly
    padded_source_image = add_borders(source_image)
    padded_labeled_source_image = add_borders(labeled_source_image)

    # create image with the billboard being the only white spot
    binary_image = strip_colours(padded_labeled_source_image)

    # identify the billboard border
    contour = get_largest_billboard_contour(binary_image)
    outlined_image = binary_image.copy()
    outlined_image = cv2.drawContours(
        outlined_image,
        [contour],
        -1,
        [0, 255, 0],
        3
    )

    # create mask for replacing billboard iamge
    blank_image = np.zeros(padded_source_image.shape, dtype=np.uint8)
    mask_image = get_contour_mask(blank_image, contour)
    mask_inv_image = cv2.bitwise_not(mask_image)

    # remove the old billboard decal
    labeled_image_no_billboard = cv2.bitwise_and(padded_labeled_source_image, mask_inv_image)
    source_image_no_billboard = cv2.bitwise_and(padded_source_image, mask_inv_image)

    # warp the decal to fit the bilboard
    warped_image = get_homo_warped_decal(outlined_image, decal_image, contour)

    # apply the decal to the billboard
    output_image = cv2.bitwise_or(source_image_no_billboard, warped_image)

    # remove border we applied earlier
    output_image_borderless = remove_borders(output_image)

    # helper image to show contour points
    circle_image = padded_labeled_source_image.copy()
    colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
    for i, center in enumerate([x[0] for x in contour]):
        cv2.circle(circle_image, tuple(center), 20, colors[i], -1)

    return output_image_borderless, contour, [
        padded_source_image,
        padded_labeled_source_image,
        binary_image,
        outlined_image,
        circle_image,
        mask_image,
        mask_inv_image,
        labeled_image_no_billboard,
        padded_source_image,
        source_image_no_billboard,
        cv2.resize(
            decal_image,
            (0, 0),
            fx=padded_source_image.shape[1] / decal_image.shape[1] * 0.5,
            fy=padded_source_image.shape[0] / decal_image.shape[0] * 0.5
        ),
        warped_image,
        output_image,
    ]

# %%
