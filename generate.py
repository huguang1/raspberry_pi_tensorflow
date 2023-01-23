#!/usr/bin/env python3

import os
import numpy
import random
import string
import cv2
import argparse
import captcha.image


def main(count, output_dir):
    width = 128
    height = 64
    length = 6
    symbols = 'symbols.txt'
    captcha_generator = captcha.image.ImageCaptcha(width=width, height=height)
    symbols_file = open(symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(output_dir):
        print("Creating output directory " + output_dir)
        os.makedirs(output_dir)

    for i in range(count):
        random_str = ''.join([random.choice(captcha_symbols) for j in range(length)])
        image_path = os.path.join(output_dir, random_str+'.png')
        if os.path.exists(image_path):
            version = 1
            while os.path.exists(os.path.join(output_dir, random_str + '_' + str(version) + '.png')):
                version += 1
            image_path = os.path.join(output_dir, random_str + '_' + str(version) + '.png')

        image = numpy.array(captcha_generator.generate_image(random_str))
        cv2.imwrite(image_path, image)


if __name__ == '__main__':
    count = 128000
    output_dir = 'train'
    main(count, output_dir)
