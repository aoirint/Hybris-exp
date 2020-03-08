OPENPOSE_MODEL_FOLDER_PATH = '/openpose/models'
OPENPOSE_INSTALL_PATH = '/usr/local/python'

OPENPOSE_NOSE = 0
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

    poses = np.asarray(datum.poseKeypoints[0], dtype=np.float32)
    keypoints = poses[:, :2]
    confs = poses[:, 2]

    return keypoints, confs, datum.cvOutputData



def split_alpha(image):
    if image.shape[2] != 4:
        h, w, c = image.shape
        A = np.full((h, w), 255, dtype=np.uint8)
        return image.copy(), A

    B, G, R, A = cv2.split(image)
    return cv2.merge([ B, G, R ]), A

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

    # http://iei.sozaiya.com/archives/12029905.html
    material_image_path = 'man_suit.png'

    target_image = cv2.imread(target_image_path)
    target_keypoints, target_confs,  target_rendered = get_keypoints(target_image)
    cv2.imwrite('target_rendered.jpg', target_rendered)

    target_leftsh_pos = target_keypoints[OPENPOSE_LEFT_SHOULDER]
    target_rightsh_pos = target_keypoints[OPENPOSE_RIGHT_SHOULDER]
    target_neck_pos = target_keypoints[OPENPOSE_NECK]
    target_nose_pos = target_keypoints[OPENPOSE_NOSE]

    material_image = cv2.imread(material_image_path, -1)
    material_bgr, material_mask = split_alpha(material_image)

    material_keypoints, material_confs, material_rendered = get_keypoints(material_bgr)
    cv2.imwrite('material_rendered.jpg', material_rendered)

    cv2.namedWindow('out', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('out', (800, 600))

    dy = 0
    while True:
        material_leftsh_pos = material_keypoints[OPENPOSE_LEFT_SHOULDER]
        material_rightsh_pos = material_keypoints[OPENPOSE_RIGHT_SHOULDER]
        material_neck_pos = material_keypoints[OPENPOSE_NECK]

        target_shoulder_distance = abs(target_leftsh_pos[0] - target_rightsh_pos[0])
        target_necknose_distance = abs(target_nose_pos[1] - target_neck_pos[1])

        material_shoulder_distance = abs(target_leftsh_pos[0] - target_rightsh_pos[0])
        material_nose_dy = target_necknose_distance / target_shoulder_distance * material_shoulder_distance
        material_nose_pos = material_neck_pos - [ 0, material_nose_dy ] - [ 0, dy ]

        target_points = np.asarray([
            target_leftsh_pos,
            target_rightsh_pos,
            # target_neck_pos,
            target_nose_pos,
        ], dtype=np.float32)
        material_points = np.asarray([
            material_leftsh_pos,
            material_rightsh_pos,
            # material_neck_pos,
            material_nose_pos,
        ], dtype=np.float32)

        M = cv2.getAffineTransform(material_points, target_points)
        # M = cv2.getPerspectiveTransform(material_points, target_points)
        # M, _ = cv2.findHomography(material_points, target_points)

        th, tw, _ = target_image.shape
        material_mask_warped = cv2.warpAffine(material_mask, M, (tw, th))
        material_bgr_warped = cv2.warpAffine(material_bgr, M, (tw, th))
        # material_mask_warped = cv2.warpPerspective(material_mask, M, (tw, th))
        # material_rgb_warped = cv2.warpPerspective(material_rgb, M, (tw, th))

        fusion_bgr = copy_to(material_bgr_warped, target_image, material_mask_warped)

        cv2.imshow('out', fusion_bgr)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('w'):
            dy -= 1
            print(dy)
        elif key & 0xFF == ord('s'):
            dy += 1
            print(dy)
        elif key & 0xFF == ord('c'):
            cv2.imwrite('out.jpg', fusion_bgr)
        elif key & 0xFF == ord('q'):
            break
