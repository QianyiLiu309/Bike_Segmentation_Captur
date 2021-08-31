import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt

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
    batch_size=BATCH_SIZE,
    class_mode=None,
    seed=seed)
mask_generator = mask_datagen.flow_from_directory(
    'dataset/train_masks',
    target_size=(128, 128),
    batch_size=BATCH_SIZE,
    class_mode=None,
    seed=seed)
train_generator = zip(image_generator, mask_generator)
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_dataset,
    output_types=(tf.float32, tf.float32),
    output_shapes=([BATCH_SIZE, 128, 128, 3], [BATCH_SIZE, 128, 128, 1])
)

test_data_gen_args = dict(featurewise_center=False,
                          featurewise_std_normalization=False)
test_image_datagen = ImageDataGenerator(**test_data_gen_args)
test_mask_datagen = ImageDataGenerator(**test_data_gen_args)
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
test_image_generator = image_datagen.flow_from_directory(
    'dataset/test_images',
    target_size=(128, 128),
    batch_size=1,
    class_mode=None,
    seed=seed)
test_mask_generator = mask_datagen.flow_from_directory(
    'dataset/test_masks',
    target_size=(128, 128),
    batch_size=1,
    class_mode=None,
    seed=seed)
test_generator = zip(image_generator, mask_generator)
test_dataset = tf.data.Dataset.from_generator(
    lambda: test_dataset,
    output_types=(tf.float32, tf.float32),
    output_shapes=([1, 128, 128, 3], [1, 128, 128, 1])
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

OUTPUT_CHANNELS = 3
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# 使用这些层的激活设置
layer_names = [
    'block_1_expand_relu',  # 64x64
    'block_3_expand_relu',  # 32x32
    'block_6_expand_relu',  # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',  # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# 创建特征提取模型
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),  # 32x32 -> 64x64
]


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs

    # 在模型中降频取样
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # 升频取样然后建立跳跃连接
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # 这是模型的最后一层
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print("model compiled")


# tf.keras.utils.plot_model(model, show_shapes=True)
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


EPOCHS = 5
VAL_SUBSPLITS = 5
VALIDATION_STEPS = 1

# model_history = model.fit(train_dataset, epochs=EPOCHS,
#                           steps_per_epoch=STEPS_PER_EPOCH,
#                           validation_steps=VALIDATION_STEPS,
#                           validation_data=test_dataset)

model_history = model.fit(train_dataset, steps_per_epoch=STEPS_PER_EPOCH,
                          epochs=5, verbose=2, shuffle=True, validation_data=test_dataset)

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()
