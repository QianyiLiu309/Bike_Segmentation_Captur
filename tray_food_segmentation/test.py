import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import os
import cv2
from unet import unet_model

tf.config.experimental.set_memory_growth = True

TRAIN_LENGTH = 1241
BATCH_SIZE = 4
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask[input_mask >= 2] = 2
    return input_image, input_mask


def load_data(img_dir_path, mask_dir_path):
    images = []
    masks = []

    for i in os.listdir(img_dir_path):
        img_path = img_dir_path + i
        im = Image.open(img_path)
        img_data = np.array(im)
        img_data = cv2.resize(img_data, (128, 128))

        mask_path = mask_dir_path + i[0:len(i) - 4] + '.png'
        mask_data = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask_data = cv2.resize(mask_data, (128, 128))
        mask_data = np.expand_dims(mask_data, -1)

        input_image, input_mask = normalize(img_data, mask_data)
        images.append(input_image)
        masks.append(input_mask)

    assert len(images) == len(masks)
    return images, masks


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


x_train, y_train = load_data('dataset/train_images/XTrain/', 'dataset/train_masks/yTrain/')
train = tf.data.Dataset.from_tensor_slices((x_train, y_train))

x_test, y_test = load_data('dataset/test_images/XTest/', 'dataset/test_masks/yTest/')
test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

print(train)
print(test)

train_dataset = train.batch(BATCH_SIZE)
# train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

print(train_dataset)
print(test_dataset)

# display((x_train[0], y_train[0]))
# display((x_test[0], y_test[0]))

OUTPUT_CHANNELS = 3
model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# tf.keras.utils.plot_model(model, show_shapes=True)


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)])
    else:
        sample_image = x_test[0]
        sample_mask = y_test[0]
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])


output = model.predict(x_test[0][tf.newaxis, ...])
print(output.shape)