import os
import cv2
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from urllib import request

from mtcnn.mtcnn import MTCNN

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms

from backbone import Backbone


def face_crop(img_idx, pixels, images, crop_folder, required_size=(112, 112)):
    """
    한 이미지에 대해 MTCNN 모델로 Face Detection 해서 crop한 사진을 crop_folder에 저장

    Args:
      img_idx (int): images[] 에서의 idx
      pixels (list): 얼굴을 detect할 이미지를 pixel list로 바꾼 것
      images (list) : img idx와 img url이 들은 list
      crop_folder (string) : crop 이미지를 저장할 path
      required_size (tuple) : default(112, 112)

    Returns:
      images (list) : [{
                      "id" : DB에서 이미지 id
                      "url" : S3에서 생성한 url
                      "group_idx" : 얼굴이 없는 사진에 대해 images["group_idx"] = -2 을 추가해 images 반환
                    }]
    """

    # FaceDetection, 얼굴탐지
    # MTCCN 모델 이용
    detector = MTCNN()
    results = detector.detect_faces(pixels)

    # 얼굴이 없는 이미지
    if len(results) == 0:
        images[img_idx]["group_idx"] = [-2]
        noFace = True

    # 얼굴이 1개이상인 이미지
    else:
        noFace = False
        crop_file_num = 1
        for i in range(len(results)):
            x1, y1, width, height = results[i]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = pixels[y1:y2, x1:x2]
            image = Image.fromarray(face)
            image = image.resize(required_size)

            # crop 폴더에 이미지 저장
            crop_path = crop_folder + str(img_idx) + "_" + str(crop_file_num) + ".jpg"
            crop_file_num = crop_file_num + 1
            #print("crop_path = ", crop_path)
            image.save(crop_path)

    return images, noFace