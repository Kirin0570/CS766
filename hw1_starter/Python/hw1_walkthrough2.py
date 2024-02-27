from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def hw1_walkthrough2():
    # Load the image "Vincent_van_Gogh.png" into memory
    img = Image.open('data/Vincent_van_Gogh.png')

    # Note the image is of the type uint8, 
    # and the maximum pixel value of the image is 255.
    print(img.mode)
    print(np.amax(img))

    # uint8 is memory efficient. Since we will perform some arithmetic operations
    # on the image, uint8 needs to be used with caution. Let's cast the image
    # to double.
    img = np.array(img, dtype=float) / 255

    print(img.dtype)
    print(np.amax(img))

    # Display the image
    plt.figure()
    plt.imshow(img)
    plt.axis(False)
    plt.show()

    # Separate the image into three color channels and store each channel into
    # a new image
    red_channel = img[:, :, 0]
    plt.figure()
    plt.imshow(red_channel, cmap='gray')
    plt.axis(False)
    plt.show()

    red_image = np.zeros(img.shape)
    red_image[:, :, 0] = red_channel
    plt.figure()
    plt.imshow(red_image)
    plt.axis(False)
    plt.show()

    # Similarly extract green_channel and blue_channel and create green_image
    # and blue_image
    # green_image = ???
    green_image = np.zeros(img.shape)
    green_image[:, :, 1] = img[:, :, 1]
    # blue_image = ???
    blue_image = np.zeros(img.shape)
    blue_image[:, :, 2] = img[:, :, 2]

    # Create a 2 x 2 image collage in the following arrangement
    # original image | red channel
    # green channel  | blue channel
    # collage_2x2 = ???
    collage_2x2 = np.zeros((img.shape[0] * 2, img.shape[1] * 2, img.shape[2]))
    # Copy the original image
    collage_2x2[:img.shape[0], :img.shape[1], :] = img
    # Copy the red channel
    collage_2x2[:img.shape[0], img.shape[1]:(2 * img.shape[1]), :] = red_image
    # Copy the green channel
    collage_2x2[img.shape[0]:(2 * img.shape[0]), :img.shape[1], :] = green_image
    # Copy the blue channel
    collage_2x2[img.shape[0]:(2 * img.shape[0]), img.shape[1]:(2 * img.shape[1]), :] = blue_image

    plt.figure()
    plt.imshow(collage_2x2)
    plt.axis(False)
    plt.show()

    # Save the collage as collage.png
    # Convert image back into uint8 (between 0 and 255) before saving
    collage_img = Image.fromarray((collage_2x2 * 255).astype(np.uint8))
    collage_img.save('outputs/collage.png')