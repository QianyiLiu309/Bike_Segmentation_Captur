import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt

# img_path = 'dataset/yTest/1005a.png'
# image = load_img(img_path)
# img_data = img_to_array(image)
# print(img_data[128])
# print(img_data.shape)
#
# img_data = np.array([img_data])
# print(img_data.shape)
# print(type(img_data))
#
# # img = Image.open(img_path)
# # img_data_2 = np.array(img)
# # img_data_2 = np.expand_dims(img_data_2, 0)
# # print(img_data_2.shape)
# # print(type(img_data_2))
#
# datagen = ImageDataGenerator(rotation_range=30, fill_mode='nearest')
#
# aug_iter = datagen.flow(img_data, batch_size=1)
#
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))
#
# for i in range(3):
#     image = next(aug_iter)
#

# train_path = 'dataset\XTrain'
# train_datagen = ImageDataGenerator(
#   width_shift_range=0.2,
#   height_shift_range=0.2
# )
# train_generator = train_datagen.flow_from_directory(r"D:\files\Cambridge\Self_learning\bike_segmentation_captur\tray_food_segmentation\dataset", target_size=(224, 224),
#                                                     class_mode=None, batch_size=1)
#
# fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 15))
#
# for i in range(4):
#
#   # convert to unsigned integers for plotting
#   image = next(train_generator)[0]
#
#   # changing size from (1, 200, 200, 3) to (200, 200, 3) for plotting the image
#   image = np.squeeze(image)
#
#   # plot raw pixel data
#   ax[i].imshow(image)
#   ax[i].axis('off')

TRAIN_LENGTH = 1241
BATCH_SIZE = 64
BUFFER_SIZE = 200
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
# image_datagen.fit(images, augment=True, seed=seed)
# mask_datagen.fit(masks, augment=True, seed=seed)
image_generator = image_datagen.flow_from_directory(
    'dataset/train_images',
    target_size=(128, 128),
    batch_size=1,
    class_mode=None,
    seed=seed)
mask_generator = mask_datagen.flow_from_directory(
    'dataset/train_masks',
    target_size=(128, 128),
    batch_size=1,
    class_mode=None,
    seed=seed)
train_generator = zip(image_generator, mask_generator)

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))
for i in range(4):
    # convert to unsigned integers for plotting
    image = next(train_generator)[1].astype('uint8')

    # changing size from (1, 200, 200, 3) to (200, 200, 3) for plotting the image
    image = np.squeeze(image)

    # plot raw pixel data
    ax[i].imshow(image)
    ax[i].axis('off')

plt.show()

train_dataset = tf.data.Dataset.from_generator(
    lambda: train_dataset,
    output_types=(tf.float32, tf.float32),
    output_shapes=([32, 128, 128, 3], [32, 128, 128, 1])
)

# fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 15))
#
# for i in range(4):
#
#   # convert to unsigned integers for plotting
#   image = next(train_generator)
#   print(image[0].shape)
#   print(image[1].shape)

train_dataset = train_dataset.apply(tf.data.experimental.assert_cardinality(1241))

print(train_dataset)
print(tf.data.experimental.cardinality(train_dataset))
print(len(train_dataset))

train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
print(train_dataset)


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


# sample_image, sample_mask = next(iter(train_dataset))
# display([sample_image, sample_mask])
