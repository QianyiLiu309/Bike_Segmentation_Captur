import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def load_image(file_path):
    img = Image.open(file_path)
    img_data = np.array(img)
    return img_data


def visualize(img_data):
    print(img_data.shape)
    plt.axis("off")
    plt.imshow(img_data)
    plt.show()


def visualize_dual(img_data_1, img_data_2):
    fig = plt.figure(figsize=(10, 6))
    rows = 1
    columns = 2

    fig.add_subplot(rows, columns, 1)
    plt.axis('off')
    plt.imshow(img_data_1)

    fig.add_subplot(rows, columns, 2)
    plt.axis('off')
    plt.imshow(img_data_2)

    plt.show()