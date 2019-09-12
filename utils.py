import matplotlib.pyplot as plt
import numpy as np
import cv2

def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')

def visualize_output(test_images, test_outputs):
    plt.figure(figsize=(20, 10))

    predicted_key_pts = test_outputs




    show_all_keypoints(np.squeeze(test_images), predicted_key_pts)

    plt.show()


def rescale(img ,v ,key_pts):
    h, w = img.shape[:2]

    if h > w:
        new_h, new_w = v * h / w, v
    else:
        new_h, new_w = v, v * w / h
    new_h, new_w = int(new_h), int(new_w)

    img = cv2.resize(img, (new_w, new_h))


    key_pts = key_pts * [new_w / w, new_h / h]


    return img ,key_pts

def random_crop(img ,v ,key_pts):
    h, w = img.shape[:2]
    new_h, new_w = (v ,v)
    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    img = img[top: top + new_h ,left: left + new_w]

    key_pts = key_pts - [left, top]

    return img ,key_pts
