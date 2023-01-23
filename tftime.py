import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import tflite_runtime.interpreter as tflite


def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])


def main(model_name, captcha_dir, output, symbols):
    symbols_file = open(symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    with open(output, 'w') as output_file:
        interpreter = tflite.Interpreter(model_path=model_name)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        for x in sorted(os.listdir(captcha_dir)):
            raw_data = cv2.imread(os.path.join(captcha_dir, x))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            image = numpy.array(rgb_data) / 255.0
            (c, h, w) = image.shape
            image = image.reshape([-1, c, h, w]).astype('float32')
            interpreter.set_tensor(input_details[0]['index'], image)
            interpreter.invoke()
            output_data_list = []
            for i in range(6):
                output_data = interpreter.get_tensor(output_details[i]['index'])
                output_data_list.append(output_data)
            output_file.write(x + ", " + decode(captcha_symbols, output_data_list) + "\n")
            print('Classified ' + x)


if __name__ == '__main__':
    main(model_name='hu.tflite', captcha_dir='pictures', output='a.csv', symbols='symbols.txt')


















