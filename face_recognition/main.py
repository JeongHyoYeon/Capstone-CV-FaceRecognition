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

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
BASE_PATH = FILE.parents[0]

if str(BASE_PATH) not in sys.path:
    sys.path.append(str(BASE_PATH))  # add BASE_PATH to PATH

from face_crop import face_crop
# from face_alignment import face_alignment
from get_embeddings import get_embeddings
from face_grouping import face_grouping


def run_face_recog(images):
    """
    FaceRecognition으로 FaceGrouping하는 main 함수.

    Args:
      images (list) : [{
                      "id" : DB에서 이미지 id
                      "url" : S3에서 생성한 url
                      "group_idx" : //여기를 완성시키는 것이 최종 목표
                    }]

    Returns:
      groups (list): [{
                      "original_images_idx_list" : 해당 crop이미지의 원본 url이 담긴 urls idx
                      "crop_path_list" : crop path
                      "face_list" : crop한 얼굴 이미지 list
                      "face2_idx_list" : all_faces의 idx, global_face_idx.
                      "face1_idx_list" : group 이미지의 어떤 이미지랑 유사도가 0.4가 넘어서 들어왔는지. (빈 리스트이면 맨 처음 추가된 이미지인것.)
                      "cosine_similarity_list" : 그 유사도가 얼마였는지 (-1 이면 맨 처음 추가된 이미지인 것)
                    }]

      group_idx_list (list) : 유효한 group의 idx만 들어있는 list. (-2, -1은 넣지 않았다)

      images (list) : [{
                      "id" : DB에서 이미지 id
                      "url" : S3에서 생성한 url
                      "group_idx" : [1, 4, 9] (해당 사진이 속한 group의 list를 전송), 얼굴이 없으면 [-2]. 얼굴이 있는데 group에 넣기엔 너무 한장일 경우 [-1].
                    }]
    """
    input_size = 112
    cos_similarity_threshold = 0.45
    # face_recog_folder = "/content/drive/MyDrive/Capstone_Face_Test/"

    crop_base_folder = os.path.join(BASE_PATH, "crop_images/")
    crop_folder = os.path.join(crop_base_folder, "crop/")

    if not os.path.isdir(crop_base_folder):
        os.mkdir(crop_base_folder)

    if os.path.exists(crop_folder):
        shutil.rmtree(crop_folder)
        os.mkdir(crop_folder)
    else:
        os.mkdir(crop_folder)

    # ----------------------------------------------------------
    #                      Step1. face detect
    # ----------------------------------------------------------
    print("Step1. face detect\n")

    folder_no_face = True

    for idx, image in enumerate(images):
        # s3에서 생성된 url을 request&response로 받아서 img로 넘기기
        res = request.urlopen(image["url"]).read()
        img = Image.open(BytesIO(res))
        img = img.convert('RGB')
        pixels = np.asarray(img)

        # face detect 얼굴 탐지
        images, file_no_face = face_crop(idx, pixels, images, crop_folder)

        if (folder_no_face == True and file_no_face == False):
            folder_no_face = False

    if (folder_no_face == True):
        return -1, -1, -1

    # ----------------------------------------------------------
    #                      Step2. embedding
    # ----------------------------------------------------------
    print("\n\n")
    print("Step2. embedding\n")

    faces, embeddings = get_embeddings(
        data_root=crop_base_folder,
        model_root=os.path.join(BASE_PATH, "checkpoint/backbone_ir50_ms1m_epoch120.pth"),
        input_size=[input_size, input_size],
    )
    # print("faces 길이 = ", len(faces))
    # print("embeddings 길이 = ", len(embeddings))

    # ----------------------------------------------------------
    #                      Step3. cos_similarity 계산
    # ----------------------------------------------------------
    print("\n\n")
    print("Step3. cos_similarity 계산\n")

    cosine_similaritys = np.dot(embeddings, embeddings.T)
    cosine_similaritys = cosine_similaritys.clip(min=0, max=1)

    # ----------------------------------------------------------
    #                      Step4. grouping
    # ----------------------------------------------------------
    print("\n\n")
    print("Step4. grouping\n")

    groups, group_idx_list, images = face_grouping(faces, images, cosine_similaritys, cos_similarity_threshold)

    return groups, group_idx_list, images
