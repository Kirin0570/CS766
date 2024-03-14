from PIL import Image, ImageDraw
import numpy as np
from typing import Union, Tuple, List


def computeHomography(src_pts_nx2: np.ndarray, dest_pts_nx2: np.ndarray) -> np.ndarray:
    '''
    Compute the homography matrix.
    Arguments:
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    Returns:
        H_3x3: the homography matrix (3x3 numpy array).
    '''

    ''' test case
    src_pts_nx2 = np.array([
        [1, 1],
        [2, 1],
        [2, 2],
        [1, 2],
        [0, -1]
    ])

    dest_pts_nx2 = np.array([
        [1.5, 2],
        [2.5, 2],
        [2.50, 3.00],
        [1.50, 3.00],
        [0.5, 0]
    ])
    '''

    n = src_pts_nx2.shape[0]

    # Compute the fisrt 3 columns of A.
    nonzeros_col1to3 = np.hstack((src_pts_nx2, np.ones((n, 1))))
    zeros_col1to3 = np.zeros((n, 3))
    # Interleave rows of the two matrices above.
    col_1to3 = np.reshape(np.hstack((nonzeros_col1to3, zeros_col1to3)), (2 * nonzeros_col1to3.shape[0], -1))

    # Compute col 4-6 similarly by interleaving.
    col_4to6 = np.reshape(np.hstack((zeros_col1to3, nonzeros_col1to3)), (2 * nonzeros_col1to3.shape[0], -1))
    
    # Compute col 7.
    xs = src_pts_nx2[:, 0]
    ys = src_pts_nx2[:, 1]
    xd = dest_pts_nx2[:, 0]
    yd = dest_pts_nx2[:, 1]

    sub_col7_xsxd = xs * xd
    sub_col7_xsyd = xs * yd
    col_7 = -np.reshape(np.vstack((sub_col7_xsxd, sub_col7_xsyd)).T, (2 * sub_col7_xsxd.size, -1))
    
    # Compute col 8.
    sub_col8_xdys = ys * xd
    sub_col8_ysyd = ys * yd
    col_8 = -np.reshape(np.vstack((sub_col8_xdys, sub_col8_ysyd)).T, (2 * sub_col8_xdys.size, -1))
    
    # Compute col 9.
    col_9 = -np.reshape(np.vstack((xd, yd)).T, (2 * xd.size, -1))
    
    # Stack the cols to obtain A.
    A = np.hstack((col_1to3, col_4to6, col_7, col_8, col_9))

    # Compute h by evd.
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.T, A))
    min_eigenvalue_index = np.argmin(eigenvalues)
    h = eigenvectors[:, min_eigenvalue_index]

    # Reshape to obtain H.
    H_3x3 = np.reshape(h, (3, 3))

    '''test
    np.dot(H_3x3, nonzeros_col1to3[0, :].T)
    '''
    
    return H_3x3



    


def applyHomography(H_3x3: np.ndarray, src_pts_nx2: np.ndarray) ->  np.ndarray:
    '''
    Apply the homography matrix to the source points.
    Arguments:
        H_3x3: the homography matrix (3x3 numpy array).
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
    Returns:
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    '''

    n = src_pts_nx2.shape[0]
    # Convert to homogeneous coordinates of src.
    nonzeros_col1to3 = np.hstack((src_pts_nx2, np.ones((n, 1))))
    # Apply H.
    dest_pts_homo = np.dot(H_3x3, nonzeros_col1to3.T).T
    # Convert back.
    last_column = dest_pts_homo[:, -1].reshape(-1, 1) 
    dest_pts_nx2 = dest_pts_homo[:, 0:2] / last_column

    return dest_pts_nx2


def showCorrespondence(img1: Image.Image, img2: Image.Image, pts1_nx2: np.ndarray, pts2_nx2: np.ndarray) -> Image.Image:
    '''
    Show the correspondences between the two images.
    Arguments:
        img1: the first image.
        img2: the second image.
        pts1_nx2: the coordinates of the points in the first image (nx2 numpy array).
        pts2_nx2: the coordinates of the points in the second image (nx2 numpy array).
    Returns:
        result: image depicting the correspondences.
    '''

    # Convert numpy arrays to PIL images
    img1_pil = Image.fromarray(img1)
    img2_pil = Image.fromarray(img2)

    # Create a new image by pasting img1 and img2 side by side
    total_width = img1_pil.width + img2_pil.width
    max_height = max(img1_pil.height, img2_pil.height)
    result = Image.new('RGB', (total_width, max_height))
    result.paste(img1_pil, (0, 0))
    result.paste(img2_pil, (img1_pil.width, 0))

    # Create a draw object to draw lines
    draw = ImageDraw.Draw(result)

    # Draw lines between corresponding points
    for (x1, y1), (x2, y2) in zip(pts1_nx2, pts2_nx2):
        # Adjust the x-coordinate of points in the second image
        x2_adj = x2 + img1_pil.width
        draw.line((x1, y1, x2_adj, y2), fill=(255, 0, 0), width=2)

    return result

# function [mask, result_img] = backwardWarpImg(src_img, resultToSrc_H, dest_canvas_width_height)

def backwardWarpImg(src_img: Image.Image, destToSrc_H: np.ndarray, canvas_shape: Union[Tuple, List]) -> Tuple[Image.Image, Image.Image]:
    '''
    Backward warp the source image to the destination canvas based on the
    homography given by destToSrc_H. 
    Arguments:
        src_img: the source image.
        destToSrc_H: the homography that maps points from the destination
            canvas to the source image.
        canvas_shape: shape of the destination canvas (height, width).
    Returns:
        dest_img: the warped source image.
        dest_mask: a mask indicating sourced pixels. pixels within the
            source image are 1, pixels outside are 0.
    '''

    dest_img = np.zeros((canvas_shape[0], canvas_shape[1], 3), dtype=np.float32)
    dest_mask = np.zeros((canvas_shape[0], canvas_shape[1]), dtype=bool)

    # Iterate over every pixel in the destination image
    for y in range(canvas_shape[0]):
        for x in range(canvas_shape[1]):
            dest_coord = np.array([x, y, 1])
            src_coord = destToSrc_H @ dest_coord
            src_coord = src_coord/src_coord[2]  # Normalize the coordinates
            
            src_x, src_y = int(src_coord[0]), int(src_coord[1])
            
            # Check if the source coordinates are within the bounds of the src_img
            if 0 <= src_x < src_img.shape[1] and 0 <= src_y < src_img.shape[0]:
                dest_img[y, x] = src_img[src_y, src_x]
                dest_mask[y, x] = True

    return dest_mask, dest_img


def blendImagePair(img1: List[Image.Image], mask1: List[Image.Image], img2: Image.Image, mask2: Image.Image, mode: str) -> Image.Image:
    '''
    Blend the warped images based on the masks.
    Arguments:
        img1: list of source images.
        mask1: list of source masks.
        img2: destination image.
        mask2: destination mask.
        mode: either 'overlay' or 'blend'
    Returns:
        out_img: blended image.
    '''
    raise NotImplementedError

def runRANSAC(src_pt: np.ndarray, dest_pt: np.ndarray, ransac_n: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Run the RANSAC algorithm to find the inliers between the source and
    destination points.
    Arguments:
        src_pt: the coordinates of the source points (nx2 numpy array).
        dest_pt: the coordinates of the destination points (nx2 numpy array).
        ransac_n: the number of iterations to run RANSAC.
        eps: the threshold for considering a point to be an inlier.
    Returns:
        inliers_id: the indices of the inliers (kx1 numpy array).
        H: the homography matrix (3x3 numpy array).
    '''
    raise NotImplementedError

def stitchImg(*args: Image.Image) -> Image.Image:
    '''
    Stitch a list of images.
    Arguments:
        args: a variable number of input images.
    Returns:
        stitched_img: the stitched image.
    '''
    raise NotImplementedError