from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import dilation, erosion
from skimage.filters import threshold_otsu


def hw2_walkthrough1():
    #----------------- 
    # Convert a grayscale image to a binary image
    #-----------------
    img = Image.open('data/coins.png')
    img = img.convert('L')  # Convert the image to grayscale
    img = np.array(img)

    # Convert the image into a binary image by applying a threshold
    # threshold = ???

    # Determine the threshold
    # Calculate the histogram of the grayscale image
    histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))
    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.plot(bin_edges[0:-1], histogram)  # bin_edges is one element longer than histogram
    plt.title('Histogram of Gray Levels')
    plt.xlabel('Gray Level')
    plt.ylabel('Number of Pixels')
    plt.grid(True)
    plt.show()

    # By checking the histogram, the value of the threshold should be about 90
    threshold = 90
    bw_img = img > threshold

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original Image')

    ax[1].imshow(bw_img, cmap='gray')
    ax[1].set_title('Binary Image')

    fig.savefig('outputs/binary_coins.png')
    plt.show()

    #----------------- 
    # Remove noises in the binary image
    #-----------------
    # Clean the image (you may notice some holes in the coins) by using
    # dilation and then erosion

    # Specify the size of the structuring element for erosion/dilation
    # k = ???

    # Tutorial: https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
    # Selecting k: I tried starting from k =1 and increased k by 1 until all holes disappeared, ending up with k = 3
    k = 3
    selem = np.ones((k, k))
    
    fig, ax = plt.subplots(1, 2)
    processed_img = dilation(bw_img, selem=np.ones((k, k)))
    ax[0].imshow(processed_img, cmap='gray')
    ax[0].set_title('After Dilation')

    # Apply erosion then dilation once to remove the noises
    processed_img = erosion(processed_img, selem=np.ones((k, k)))
    ax[1].imshow(processed_img, cmap='gray')
    ax[1].set_title('After Erosion')

    plt.savefig('outputs/noise_removal_coins.png')
    plt.show()

    #----------------- 
    # Remove the rices
    #-----------------
    # Apply erosion then dilation once to remove the rices

    # Specify the size of the structuring element for erosion/dilation
    # k = ???
    # Selecting k: I tried starting from k = 20 and increased k by 1 until all holes disappeared, ending up with k = 23
    k = 23
    selem = np.ones((k, k))

    fig, ax = plt.subplots(1, 2)
    processed_img = erosion(processed_img, selem=selem)
    ax[0].imshow(processed_img, cmap='gray')
    ax[0].set_title('After Erosion')

    processed_img = dilation(processed_img, selem=np.ones((k, k)))
    ax[1].imshow(processed_img, cmap='gray')
    ax[1].set_title('After Dilation')

    fig.savefig('outputs/morphological_operations_coins.png')
    plt.show()
