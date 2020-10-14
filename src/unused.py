
# noinspection PyShadowingNames
def get_perspective_warped_decal(background, decal, contour):
    src = np.float32([
        [0,0],
        [decal.shape[0], 0],
        decal.shape[:2],
        [0, decal.shape[1]],
    ])
    dst = np.float32(contour)
    m = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(
        decal_image,
        m,
        (background.shape[1], background.shape[0])
    )
    return warped

# noinspection PyShadowingNames
def get_affine_warped_decal(background, decal, contour):
    src = np.float32([
        [0,0],
        [decal.shape[0], 0],
        # decal.shape[:2],
        [0, decal.shape[1]],
    ])
    dst = np.float32([x[0] for i,x in enumerate(contour) if i!=2])
    m = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(
        decal_image,
        m,
        (background.shape[1], background.shape[0])
    )
    return warped
