from PIL import Image, ImageDraw
import numpy as np

def generateHoughAccumulator(edge_image: np.ndarray, theta_num_bins: int, rho_num_bins: int) -> np.ndarray:
    '''
    Generate the Hough accumulator array.
    Arguments:
        edge_image: the edge image.
        theta_num_bins: the number of bins in the theta dimension.
        rho_num_bins: the number of bins in the rho dimension.
    Returns:
        hough_accumulator: the Hough accumulator array.
    '''

    A = np.zeros((theta_num_bins, rho_num_bins))

    nrow, ncol = edge_image.shape
    diag_len = np.sqrt(nrow**2 + ncol**2)

    for i in range(nrow):
        for j in range(ncol):
            if edge_image[i, j] > 0:
                for k in range(theta_num_bins):
                    theta = k * np.pi/theta_num_bins
                    rho = j * np.cos(theta) - i * np.sin(theta)
                    # compute the bin index for rho
                    l = np.floor((rho + diag_len)/(2 * diag_len/rho_num_bins))
                    if l >= 0 and l < rho_num_bins:
                        A[k, int(l)] += 1

    min_val = np.min(A)
    max_val = np.max(A)
    normalized_A = np.ceil((A - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    return normalized_A



def lineFinder(orig_img: np.ndarray, hough_img: np.ndarray, hough_threshold: float):
    '''
    Find the lines in the image.
    Arguments:
        orig_img: the original image.
        hough_img: the Hough image.
        hough_threshold: the threshold for the Hough accumulator array.
    Returns: 
        line_img: PIL image with lines drawn.

    '''

    hough_peaks = np.where(hough_img > hough_threshold)

    line_img = Image.fromarray(orig_img.astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(line_img) 

    theta_num_bins, rho_num_bins = hough_img.shape
    nrow, ncol = orig_img.shape
    diag_len = np.sqrt(nrow**2 + ncol**2)

    for i, j in zip(*hough_peaks):
        theta = i * np.pi/theta_num_bins
        rho = -diag_len + j * (2 * diag_len/rho_num_bins)
        xp0 = 0; yp0 = rho/np.cos(theta)
        xp1 = nrow - 1; yp1 = (rho + xp1 * np.sin(theta))/np.cos(theta)
        draw.line((yp0, xp0, yp1, xp1), fill=128, width=5)
    
    line_img.show()
    return  line_img

def lineSegmentFinder(orig_img: np.ndarray, edge_img: np.ndarray, hough_img: np.ndarray, hough_threshold: float):
    '''
    Find the line segments in the image.
    Arguments:
        orig_img: the original image.
        edge_img: the edge image.
        hough_img: the Hough image.
        hough_threshold: the threshold for the Hough accumulator array.
    Returns:
        line_segement_img: PIL image with line segments drawn.
    '''
    raise NotImplementedError
