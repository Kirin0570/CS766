    obj_db = np.load('outputs/obj_db_1.npy')
    img_list = ['two_objects.png', 'many_objects_2.png']

    for i in range(len(img_list)):
        labeled_img = Image.open(f'outputs/labeled_{img_list[i]}')
        labeled_img = np.array(labeled_img)
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.

        recognizeObjects(orig_img, labeled_img, obj_db,
                         f'outputs/testing1c_2_{img_list[i]}')  