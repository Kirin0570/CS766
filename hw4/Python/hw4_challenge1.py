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


def showCorrespondence(img1: np.ndarray, img2: np.ndarray, pts1_nx2: np.ndarray, pts2_nx2: np.ndarray) -> Image.Image:
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
    for (x1, y1), (x2, y2) in zip(pts1_nx2, pts2_nx2):  # x: col index    y: row index
        # Adjust the x-coordinate of points in the second image
        x2_adj = x2 + img1_pil.width
        draw.line((x1, y1, x2_adj, y2), fill=(255, 0, 0), width=2)

    return result

# function [mask, result_img] = backwardWarpImg(src_img, resultToSrc_H, dest_canvas_width_height)

def backwardWarpImg(src_img: np.ndarray, destToSrc_H: np.ndarray, canvas_shape: Union[Tuple, List]) -> Tuple[Image.Image, Image.Image]:
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


def blendImagePair(img1: np.ndarray, mask1: np.ndarray, img2: np.ndarray, mask2: np.ndarray, mode: str) -> np.ndarray:
    '''
    Blend the images based on the masks without altering the original images.
    Arguments:
        img1: Source image as a NumPy array (RGB).
        mask1: Source mask as a NumPy array (single channel).
        img2: Destination image as a NumPy array (RGB).
        mask2: Destination mask as a NumPy array (single channel).
        mode: Either 'overlay' or 'blend'.
    Returns:
        out_img: Blended image as a NumPy array (RGB).
    '''
    
    # Create a copy of img1 to ensure the original is not modified
    out_img = np.copy(img1)

    # Ensure masks are boolean arrays
    mask1_bool = mask1 > 0
    mask2_bool = mask2 > 0

    if mode == 'overlay':
        # For overlay, directly copy pixels from img2 to the copy of img1 where mask2 applies
        out_img[mask2_bool] = img2[mask2_bool]
    elif mode == 'blend':
        # For blending, calculate the blend where both masks are positive
        both_masks = mask1_bool & mask2_bool
        only_mask1 = mask1_bool & ~mask2_bool
        only_mask2 = mask2_bool & ~mask1_bool
        
        # No need to initialize out_img with zeros since it's already a copy of img1
        out_img[both_masks] = (out_img[both_masks].astype('float') + img2[both_masks].astype('float')) / 2
        # The following lines are not needed since out_img is already img1 where only mask1 applies
        # out_img[only_mask1] = out_img[only_mask1]
        out_img[only_mask2] = img2[only_mask2]
    else:
        raise ValueError("Invalid blending mode. Choose either 'overlay' or 'blend'.")

    return out_img




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

    best_inliers_id = []
    best_H = None
    num_points = src_pt.shape[0]

    for _ in range(ransac_n):
        # Randomly select 4 points
        indices = np.random.choice(num_points, 4, replace=False)
        src_sample = src_pt[indices]
        dest_sample = dest_pt[indices]

        # Estimate homography
        H = computeHomography(src_sample, dest_sample)

        # Transform src_pt to dest plane
        transformed_pts = applyHomography(H, src_pt)

        # Calculate distances
        differences = transformed_pts - dest_pt
        distances = np.sqrt(np.sum(differences**2, axis=1))

        # Determine inliers
        inliers_id = np.where(distances < eps)[0]

        if len(inliers_id) > len(best_inliers_id):
            best_inliers_id = inliers_id
            best_H = H

    return best_inliers_id, best_H
    



def stitchImg(*args: np.ndarray) -> np.ndarray:
    '''
    Stitch a list of images.
    Arguments:
        args: a variable number of input images.
    Returns:
        stitched_img: the stitched image.
    '''
    center = int(len(args)/2)
    NB = np.array(args[center])
    leftToNB = np.eye(3)
    rightToNB = np.eye(3)

    for dev in range(1, center + 1):
        i = center - dev
        NB, leftToNB, rightToNB = stitchOneMore(currentBase = NB, leftToCB = leftToNB, rightToCB = rightToNB, img_dst = args[i + 1], img_src = args[i], left = True)
        i = center + dev
        if i < len(args):
            NB, leftToNB, rightToNB = stitchOneMore(currentBase = NB, leftToCB = leftToNB, rightToCB = rightToNB, img_dst = args[i - 1], img_src = args[i], left = False)
    
    return NB






def stitchOneMore(currentBase: np.ndarray, leftToCB: np.ndarray, rightToCB: np.ndarray, img_dst: np.ndarray, img_src: np.ndarray, left: bool):
    '''
    Stitch one more image.
    Arguments:
        currentBase: current base image (CB).
        leftToCB: the homography of the left-most stitched image to CB.
        rightToCB: the homography of the left-most stitched image to CB.
        img_dst: neighbor to img_src.
        img_src: image to be stitched.
        left: stitch on the left end or not.

    Returns:
        result: the stitched image.
    '''
    if left:
        dstToCB = leftToCB
    else:
        dstToCB = rightToCB
    from helpers import genSIFTMatches
    from hw4_challenge1 import showCorrespondence, runRANSAC

    xs, xd = genSIFTMatches(img_src, img_dst)

    reordered_xs = xs[:, [1, 0]]
    reordered_xd = xd[:, [1, 0]]

    # Use RANSAC to reject outliers
    ransac_n = 30  # Max number of iterations
    ransac_eps = 1  # Acceptable alignment error 

    inliers_id, _ = runRANSAC(xs, xd, ransac_n, ransac_eps)
    src_pts = reordered_xs[inliers_id, :]
    dest_pts = reordered_xd[inliers_id, :]


    # Copy from challenge1a
    from hw4_challenge1 import computeHomography, applyHomography, backwardWarpImg, blendImagePair
    srcToDst = computeHomography(np.array(src_pts), np.array(dest_pts))
    srcToCB = np.dot(dstToCB, srcToDst)

    # Compute the size of the new base image
    height, width = img_src.shape[:2]
    corners_src = np.array([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])
    cornersInCB = applyHomography(srcToCB, corners_src)

    left_boundary = min(cornersInCB[:, 0])
    left_boundary = min(left_boundary, 0)
    left_boundary = int(np.floor(left_boundary))

    right_boundary = max(cornersInCB[:, 0])
    right_boundary = max(right_boundary, currentBase.shape[1])
    right_boundary = int(np.ceil(right_boundary))

    top_boundary = min(cornersInCB[:, 1])
    top_boundary = min(top_boundary, 0)
    top_boundary = int(np.floor(top_boundary))

    bottom_boundary = max(cornersInCB[:, 1])
    bottom_boundary = max(bottom_boundary, currentBase.shape[0])
    bottom_boundary = int(np.ceil(bottom_boundary))


    new_base = np.zeros((bottom_boundary - top_boundary, right_boundary - left_boundary, 3))
    # Place currentBase
    new_base[(- top_boundary):(- top_boundary + currentBase.shape[0]), (- left_boundary):(- left_boundary + currentBase.shape[1])] = currentBase
    mask2 = np.any(new_base > 0, axis=2)

    # Place img_src
    # Copy from challenge 1b
    dest_canvas_shape = new_base.shape[:2]
    NBToCB = np.array([[1, 0, left_boundary], [0, 1, top_boundary], [0, 0, 1]])
    CBToDst = np.linalg.inv(dstToCB)
    dstToSrc = np.linalg.inv(srcToDst)
    NBToSrc = dstToSrc @ CBToDst @ NBToCB
    mask1, dest_img = backwardWarpImg(img_src, NBToSrc, dest_canvas_shape)
    # Superimpose the image
    result = blendImagePair(dest_img, mask1, new_base, mask2, "blend")
    result = Image.fromarray((result).astype(np.uint8))

    leftToNB = leftToCB @ np.linalg.inv(NBToCB)
    rightToNB = rightToCB @ np.linalg.inv(NBToCB)

    return np.array(result), leftToNB, rightToNB





