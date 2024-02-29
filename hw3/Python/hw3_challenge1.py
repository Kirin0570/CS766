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

    '''
    edge_image = Image.open('outputs/edge_hough_1.png')
    edge_image = np.array(edge_image.convert('L'))  # Convert the image to grayscale
    plt.imshow(edge_image, cmap='gray')
    plt.axis('off')
    plt.show()

    theta_num_bins = 30
    rho_num_bins = 30
    '''

    A = np.zeros((theta_num_bins, rho_num_bins))

    nrow, ncol = edge_image.shape
    diag_len = np.sqrt(nrow**2 + ncol**2)

    for i in range(nrow):
        for j in range(ncol):
            if edge_image[i, j] > 0:
                x = j
                y = nrow - 1 - i
                for k in range(theta_num_bins):
                    theta = k * np.pi/theta_num_bins
                    rho = y * np.cos(theta) - x * np.sin(theta)
                    # compute the bin index for rho
                    l = np.floor((rho + diag_len)/(2 * diag_len/rho_num_bins))
                    if l >= 0 and l < rho_num_bins:
                        A[k, int(l)] += 1
    
    return A



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

    '''
    orig_img = Image.open('data/hough_1.png')
    orig_img = np.array(orig_img.convert('L'))
    hough_img = Image.open('outputs/accumulator_hough_1.png')
    hough_img = np.array(hough_img.convert('L'))
    plt.imshow(edge_image, cmap='gray')
    plt.axis('off')
    plt.show()

    hough_threshold = 300
    '''
    hough_threshold = 250
    hough_peaks = np.where(hough_img > hough_threshold)
    #print(hough_peaks)

    line_img = Image.fromarray(orig_img.astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(line_img) 

    theta_num_bins, rho_num_bins = hough_img.shape
    nrow, ncol = orig_img.shape
    diag_len = np.sqrt(nrow**2 + ncol**2)

    for i, j in zip(*hough_peaks):
        theta = i * np.pi/theta_num_bins
        rho = -diag_len + j * (2 * diag_len/rho_num_bins)
        xp0 = 0; yp0 = rho/np.cos(theta)
        xp1 = ncol - 1; yp1 = (rho + xp1 * np.sin(theta))/np.cos(theta)
        draw.line((xp0, nrow - 1 - yp0, xp1, nrow - 1 -yp1), fill=128, width=3)
    
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
