import sys
sys.path.append('/usr/local/python')

import cv2
from openpose import pyopenpose as op

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str)
    args = parser.parse_args()

    MODEL_PATH = '/openpose/models'

    params = {
        'model_folder': MODEL_PATH,
    }
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    image_path = args.image
    datum = op.Datum()
    img = cv2.imread(image_path)
    datum.cvInputData = img

    opWrapper.emplaceAndPop([datum])
    cv2.imwrite('out.jpg', datum.cvOutputData)
