import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from PIL import Image
import numpy as np

from skimage.measure import label


def generateLabeledImage(gray_img: np.ndarray, threshold: float) -> np.ndarray:
    '''
    Generates a labeled image from a grayscale image by assigning unique labels to each connected component.
    Arguments:
        gray_img: grayscale image.
        threshold: threshold for the grayscale image.
    Returns:
        labeled_img: the labeled image.
    '''
    bw_img = gray_img > threshold
    labeled_img = label(bw_img, background=0)

    return labeled_img

def compute2DProperties(orig_img: np.ndarray, labeled_img: np.ndarray) ->  np.ndarray:
    '''
    Compute the 2D properties of each object in labeled image.
    Arguments:
        orig_img: the original image.
        labeled_img: the labeled image.
    Returns:
        obj_db: the object database, where each row contains the properties
            of one object.
    '''

    objects = np.unique(labeled_img)[1:]  # exclude background label (0)
    obj_db = []

    for obj_label in objects:
        # Find the pixels belonging to the current object
        coords = np.column_stack(np.where(labeled_img == obj_label))

        # Compute the centroid (mean for row and column)
        x_bar, y_bar = coords.mean(axis=0)

        # Compute the orientation
        a_prime, c_prime = np.sum(np.square(coords), axis=0)
        b_prime = 2 * np.sum(coords[:, 0] * coords[:, 1])

        A = coords.shape[0]
        a = a_prime - x_bar**2 * A
        c = c_prime - y_bar**2 * A
        b = b_prime - 2 * x_bar* y_bar * A

        theta1 = np.arctan2(b, a - c) / 2
        orientation = np.rad2deg(theta1)

        # Compute the minimum moment of	inertia (Emin) and roundness
        theta2 = theta1 + np.pi / 2
        Emin = a * np.square(np.sin(theta1)) - b * np.sin(theta1) * np.cos(theta1) + c * np.square(np.cos(theta1))
        Emax = a * np.square(np.sin(theta2)) - b * np.sin(theta2) * np.cos(theta2) + c * np.square(np.cos(theta2))
        roundness = Emin / Emax

        # Store the computed properties
        obj_db.append([obj_label, x_bar, y_bar, Emin, orientation, roundness])

    return np.array(obj_db)

def recognizeObjects(orig_img: np.ndarray, labeled_img: np.ndarray, obj_db: np.ndarray, output_fn: str):
    '''
    Recognize the objects in the labeled image and save recognized objects to output_fn
    Arguments:
        orig_img: the original image.
        labeled_img: the labeled image.
        obj_db: the object database, where each row contains the properties 
            of one object.
        output_fn: filename for saving output image with the objects recognized.
    '''

    # Create the object database for the oirg_img/labeled_img
    obj_db_ag  = compute2DProperties(orig_img, labeled_img)

    # Matching
    threshold = 0.1
    diff_abs = np.abs(obj_db[:, -1][:, np.newaxis] - obj_db_ag[:, -1])
    means = (obj_db[:, -1][:, np.newaxis] + obj_db_ag[:, -1]) / 2
    means[means == 0] = np.finfo(float).eps  # Avoid dividing 0. Replace with a small value
    relative_diff = diff_abs / means
    condition = np.any(relative_diff <= threshold, axis=0)
    recognized_obj = obj_db_ag[condition]

    # Visualization
    fig, ax = plt.subplots()
    plt.axis(False)
    ax.imshow(orig_img, cmap='gray')
    for i in range(recognized_obj.shape[0]):
        # plot the position
        ax.plot(recognized_obj[i, 2], recognized_obj[i, 1], 'rs', markerfacecolor='w')
        # plot the orientation
        theta1 = np.deg2rad(recognized_obj[i, 4])
        ax.arrow(recognized_obj[i, 2], recognized_obj[i, 1], 50 * np.sin(theta1), 50 * np.cos(theta1), head_width=3, head_length=10, fc='r', ec='r')
    plt.savefig(output_fn)
    plt.show()



def hw2_challenge1a():
    import matplotlib.cm as cm
    from skimage.color import label2rgb
    img_list = ['two_objects.png', 'many_objects_1.png', 'many_objects_2.png']
    threshold_list = [0.5, 0.5, 0.5]   # You need to find the right thresholds
    # By checking the histograms, I found the threshold should be 0.5.

    for i in range(len(img_list)):
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.
        labeled_img = generateLabeledImage(orig_img, threshold_list[i])
        Image.fromarray(labeled_img.astype(np.uint8)).save(
            f'outputs/labeled_{img_list[i]}')
        
        cmap = np.array(cm.get_cmap('Set1').colors)
        rgb_img = label2rgb(labeled_img, colors=cmap, bg_label=0)
        Image.fromarray((rgb_img * 255).astype(np.uint8)).save(
            f'outputs/rgb_labeled_{img_list[i]}')

def hw2_challenge1b():
    labeled_two_obj = Image.open('outputs/labeled_two_objects.png')
    labeled_two_obj = np.array(labeled_two_obj)
    orig_img = Image.open('data/two_objects.png')
    orig_img = np.array(orig_img.convert('L')) / 255.
    obj_db  = compute2DProperties(orig_img, labeled_two_obj)
    np.save('outputs/obj_db.npy', obj_db)
    print(obj_db)
    
    # TODO: Plot the position and orientation of the objects
    # Use a dot or star to annotate the position and a short line segment originating from the dot for orientation
    # Refer to demoTricksFun.py for examples to draw dots and lines. 
    fig, ax = plt.subplots()
    plt.axis(False)
    ax.imshow(orig_img, cmap='gray')
    for i in range(obj_db.shape[0]):
        # plot the position
        ax.plot(obj_db[i, 2], obj_db[i, 1], 'rs', markerfacecolor='w')
        # plot the orientation
        theta1 = np.deg2rad(obj_db[i, 4])
        ax.arrow(obj_db[i, 2], obj_db[i, 1], 50 * np.sin(theta1), 50 * np.cos(theta1), head_width=3, head_length=10, fc='r', ec='r')
    plt.savefig('outputs/two_objects_properties.png')
    plt.show()



def hw2_challenge1c():
    obj_db = np.load('outputs/obj_db.npy')
    img_list = ['many_objects_1.png', 'many_objects_2.png']

    for i in range(len(img_list)):
        labeled_img = Image.open(f'outputs/labeled_{img_list[i]}')
        labeled_img = np.array(labeled_img)
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.

        recognizeObjects(orig_img, labeled_img, obj_db,
                         f'outputs/testing1c_{img_list[i]}')
        
    # Use many_objects_1.png as the reference database
    # Create the object database for many_objects_1.png
    labeled_img = Image.open('outputs/labeled_many_objects_1.png')
    labeled_img = np.array(labeled_img)
    orig_img = Image.open('data/many_objects_1.png')
    orig_img = np.array(orig_img.convert('L')) / 255.
    obj_db  = compute2DProperties(orig_img, labeled_img)
    
    img_list = ['two_objects.png', 'many_objects_2.png']

    for i in range(len(img_list)):
        labeled_img = Image.open(f'outputs/labeled_{img_list[i]}')
        labeled_img = np.array(labeled_img)
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.

        recognizeObjects(orig_img, labeled_img, obj_db,
                         f'outputs/testing1c_2_{img_list[i]}')    
