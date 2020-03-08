OPENPOSE_MODEL_FOLDER_PATH = '/openpose/models'
OPENPOSE_INSTALL_PATH = '/usr/local/python'

OPENPOSE_NECK = 1
OPENPOSE_LEFT_SHOULDER = 5
OPENPOSE_RIGHT_SHOULDER = 2

import sys
sys.path.append(OPENPOSE_INSTALL_PATH)

import numpy as np
import cv2
from openpose import pyopenpose as op

def get_keypoints(image):
    datum = op.Datum()
    datum.cvInputData = image

    opWrapper.emplaceAndPop([datum])

    left_shoulder = datum.poseKeypoints[0][OPENPOSE_LEFT_SHOULDER]
    right_shoulder = datum.poseKeypoints[0][OPENPOSE_RIGHT_SHOULDER]
    neck_shoulder = datum.poseKeypoints[0][OPENPOSE_NECK]

    left_pos, left_conf = np.asarray(left_shoulder[:2]), left_shoulder[2]
    right_pos, right_conf = np.asarray(right_shoulder[:2]), right_shoulder[2]
    neck_pos, neck_conf = np.asarray(neck_shoulder[:2]), neck_shoulder[2]

    return np.asarray([left_pos, right_pos, neck_pos]), datum.cvOutputData

def split_rgba(image):
    B, G, R, A = cv2.split(image)
    return cv2.merge([ R, G, B ]), A

def copy_to(source, dest, mask):
    mask_1ch = np.expand_dims(mask, axis=2) != 0
    return np.uint8(source * mask_1ch + dest * (1 - mask_1ch))

if __name__ == '__main__':
    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('image', type=str)
    # args = parser.parse_args()

    params = {
        'model_folder': OPENPOSE_MODEL_FOLDER_PATH,
    }
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # target_image_path = args.image
    # https://www.pakutaso.com/20161057291post-9287.html
    target_image_path = '00_PP04_PP_TP_V.jpg'

    # http://iei.sozaiya.com/archives/cat_303519.html
    material_image_path = 'man_suit.png'

    target_image = cv2.imread(target_image_path)
    target_points, target_rendered = get_keypoints(target_image)
    cv2.imwrite('target_rendered.jpg', target_rendered)

    material_image = cv2.imread(material_image_path, -1)
    material_rgb, material_mask = split_rgba(material_image)

    material_points, material_rendered = get_keypoints(material_rgb)
    cv2.imwrite('material_rendered.jpg', material_rendered)

    affine = cv2.getAffineTransform(material_points, target_points)

    th, tw, _ = target_image.shape
    material_mask_warped = cv2.warpAffine(material_mask, affine, (tw, th))
    material_rgb_warped = cv2.warpAffine(material_rgb, affine, (tw, th))

    fusion_rgb = copy_to(material_rgb_warped, target_image, material_mask_warped)

    fusion = cv2.cvtColor(fusion_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite('out.jpg', fusion)
