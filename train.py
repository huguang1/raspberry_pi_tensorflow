import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
import tensorflow as tf
import tensorflow.keras as keras


def create_model(captcha_length, captcha_num_symbols, input_shape, model_depth=5, module_size=2):
  input_tensor = keras.Input(input_shape)
  x = input_tensor
  for i, module_length in enumerate([module_size] * model_depth):
      for j in range(module_length):
          x = keras.layers.Conv2D(32*2**min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
          x = keras.layers.BatchNormalization()(x)
          x = keras.layers.Activation('relu')(x)
      x = keras.layers.MaxPooling2D(2)(x)

  x = keras.layers.Flatten()(x)
  x = [keras.layers.Dense(captcha_num_symbols, activation='softmax', name='char_%d'%(i+1))(x) for i in range(captcha_length)]
  model = keras.Model(inputs=input_tensor, outputs=x)

  return model


class ImageSequence(keras.utils.Sequence):
    def __init__(self, directory_name, batch_size, captcha_length, captcha_symbols, captcha_width, captcha_height):
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.captcha_length = captcha_length
        self.captcha_symbols = captcha_symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height

        file_list = os.listdir(self.directory_name)
        self.files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.used_files = []
        self.count = len(file_list)

    def __len__(self):
        return int(numpy.floor(self.count / self.batch_size))

    def __getitem__(self, idx):
        X = numpy.zeros((self.batch_size, self.captcha_height, self.captcha_width, 3), dtype=numpy.float32)
        y = [numpy.zeros((self.batch_size, len(self.captcha_symbols)), dtype=numpy.uint8) for i in range(self.captcha_length)]

        for i in range(self.batch_size):
            if len(list(self.files.keys())) == 0:
                break
            random_image_label = random.choice(list(self.files.keys()))
            random_image_file = self.files[random_image_label]

            # We've used this image now, so we can't repeat it in this iteration
            self.used_files.append(self.files.pop(random_image_label))

            # We have to scale the input pixel values to the range [0, 1] for
            # Keras so we divide by 255 since the image is 8-bit RGB
            raw_data = cv2.imread(os.path.join(self.directory_name, random_image_file))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            processed_data = numpy.array(rgb_data) / 255.0
            X[i] = processed_data

            # We have a little hack here - we save captchas as TEXT_num.png if there is more than one captcha with the text "TEXT"
            # So the real label should have the "_num" stripped out.

            random_image_label = random_image_label.split('_')[0]

            for j, ch in enumerate(random_image_label):
                y[j][i, :] = 0
                y[j][i, self.captcha_symbols.find(ch)] = 1

        return X, y


def main(batch_size, epochs, output_model_name, train_dataset, validate_dataset):
    width = 128
    height = 64
    length = 6
    symbols = 'symbols.txt'
    input_model = None

    captcha_symbols = None
    with open(symbols) as symbols_file:
        captcha_symbols = symbols_file.readline()

    with tf.device('/device:CPU:0'):
        model = create_model(length, len(captcha_symbols), (height, width, 3))

        if input_model is not None:
            model.load_weights(input_model)

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                      metrics=['accuracy'])
        model.summary()
        training_data = ImageSequence(train_dataset, batch_size, length, captcha_symbols, width, height)
        validation_data = ImageSequence(validate_dataset, batch_size, length, captcha_symbols, width, height)

        callbacks = [keras.callbacks.EarlyStopping(patience=3),
                     keras.callbacks.ModelCheckpoint(output_model_name+'.h5', save_best_only=False)]

        with open(output_model_name+".json", "w") as json_file:
            json_file.write(model.to_json())

        try:
            model.fit_generator(generator=training_data,
                                validation_data=validation_data,
                                epochs=epochs,
                                callbacks=callbacks,
                                use_multiprocessing=True)
        except KeyboardInterrupt:
            print('KeyboardInterrupt caught, saving current weights as ' + output_model_name+'_resume.h5')
            model.save_weights(output_model_name+'_resume.h5')


if __name__ == '__main__':
    batch_size = 32
    epochs = 5
    output_model_name = 'hu'
    train_dataset = 'train'
    validate_dataset = 'validate'
    main(batch_size, epochs, output_model_name, train_dataset, validate_dataset)
