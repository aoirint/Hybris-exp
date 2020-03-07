OPENPOSE_MODEL_FOLDER_PATH = '/openpose/models'
OPENPOSE_INSTALL_PATH = '/usr/local/python'

import sys
sys.path.append(OPENPOSE_INSTALL_PATH)

import cv2
from openpose import pyopenpose as op

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str)
    args = parser.parse_args()

    params = {
        'model_folder': OPENPOSE_MODEL_FOLDER_PATH,
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
