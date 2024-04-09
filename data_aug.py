import os
import numpy as np
import cv2

def flip_images_and_bboxes(root_dir, annotations_file, save_dir, save_annotations_file, start_index=253):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    annotations = np.loadtxt(annotations_file, delimiter=',')
    image_files = [os.path.join(root_dir, f'{index + 1:08}.jpg') for index in range(len(annotations))]

    # prepare to save adjusted bounding boxes
    adjusted_annotations = []

    for i, image_path in enumerate(image_files):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping file {image_path}, unable to read.")
            continue

        flipped_image = cv2.flip(image, 1)
        
        # generate new filename starting from specified index
        new_filename = f"{start_index + i:08}.jpg"
        save_path = os.path.join(save_dir, new_filename)
        
        cv2.imwrite(save_path, flipped_image)
        print(f"Saved flipped image to {save_path}")

        #adjust bounding box annotations for the flipped image
        W = image.shape[1]  # Image width
        xmin, ymin, xmax, ymax = annotations[i]
        new_xmin = W - xmax
        new_xmax = W - xmin

        # save the adjusted bounding boxes
        adjusted_annotations.append([new_xmin, ymin, new_xmax, ymax])

    # save the adjusted annotations to a new file
    np.savetxt(save_annotations_file, adjusted_annotations, delimiter=',', fmt='%f')
    print(f"Saved adjusted bounding boxes to {save_annotations_file}")



root_dir = 'data'
annotations_file = 'data/annotations.txt'
save_dir = 'validation'
save_annotations_file = 'validation/annotations.txt'

flip_images_and_bboxes(root_dir, annotations_file, save_dir, save_annotations_file, start_index=253)
