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

from face_recognition.backbone import Backbone

#----------------------------------------------------------
#                        functions 
#----------------------------------------------------------
def face_crop(img_idx, pixels, images, crop_folder, url_dict, required_size=(112, 112)): 
    """
    한 이미지에 대해 MTCNN 모델로 Face Detection 해서 crop한 사진을 crop_folder에 저장

    Args:
      img_idx (int): images[] 에서의 idx
      pixels (list): 얼굴을 detect할 이미지를 pixel list로 바꾼 것
      images (list) : img idx와 img url이 들은 list
      crop_folder (string) : crop 이미지를 저장할 path
      url_dict (dictionary) : crop img path로 original url을 찾을 수 있게 만든 dict. key = crop image path, value = images idx
      required_size (tuple) : default(112, 112)

    Returns:
      images (list) : [{
                      "id" : DB에서 이미지 id
                      "url" : S3에서 생성한 url
                      "group_idx" : 얼굴이 없는 사진에 대해 images["group_idx"] = -2 을 추가해 images 반환
                    }]

      url_dict (dictionary) : crop img path로 original url을 찾을 수 있게 만든 dict. key = crop image path, value = images idx
    """

    # FaceDetection, 얼굴탐지
    # MTCCN 모델 이용
    detector = MTCNN()
    results = detector.detect_faces(pixels)

    # 얼굴이 없는 이미지
    if len(results)==0:
      images[img_idx]["group_idx"] = [-2]

    # 얼굴이 1개이상인 이미지 
    crop_file_num = 1
    for i in range(len(results)):
        x1, y1, width, height = results[i]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)

        # crop 폴더에 이미지 저장
        crop_path = crop_folder + '/' + str(img_idx) + "_" + str(crop_file_num) + ".jpg" 
        crop_file_num = crop_file_num + 1
        print("crop_path = ", crop_path)
        image.save(crop_path)

        # dictionary를 이용해 crop_path와 original_image_idx mapping 
        url_dict[crop_path] = img_idx

    return images, url_dict


def alignment(): 
    """
    얼굴 각도 조절하는 함수 (미완성)
    robust하게 하려면 alignment 진행해야하지만
    이거까지 넣으면 속도가 너무 느려질 것 같아서 일부러 안넣음

    Args:
      

    Returns:
      

    Details:
  
    """


def get_embeddings(data_root, model_root, input_size=[112, 112], embedding_size=512):
    """
    data root에 들은 모든 이미지들을 Arcface 모델을 이용해 embedding하는 함수

    Args:
      data_root (str) : crop된 이미지들이 들어있는 폴더의 path (이 폴더 안에 하위폴더가 한개이상 존재해야함 ex)data_root/cat, data_root/dog 폴더 구조 처럼)
      model_root (str) : arcface 모델의 .pth 파일의 path
      input_size (list) : default [112, 112]
      embedding_size (int) : default 512

    Returns:
      faces (list): [{
                      "crop_path" : crop_path
                      "face_idx" : 전체 얼굴 중에서 몇번째 얼굴인지 (이게 all_faces의 idx랑 동일함)
                      "face" : [얼굴 pixel RGB] 
                    }]

                    cf) faces에 들어있는 순서랑 embeddings에 들어있는 순서를 맞춰야해서 faces를 face_crop()아니고 get_embeddings()에서 호출함.

      embeddings (list):[[embedding1], [embedding2] ... [embeddingN]]
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data & model 경로
    assert os.path.exists(data_root)
    assert os.path.exists(model_root)
    print(f"Data root: {data_root}")

    # define image preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize(
                [int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)],
            ),  # smaller side resized
            transforms.CenterCrop([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ],
    )

    # define data loader
    dataset = datasets.ImageFolder(data_root, transform)
    loader = data.DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0,
    )

    # chekpoint에서 backbone weight 불러오기
    backbone = Backbone(input_size)
    backbone.load_state_dict(torch.load(model_root, map_location=torch.device("cpu")))
    backbone.to(device)
    backbone.eval()

    # crop 폴더에 들은 얼굴마다 embedding 하기
    embeddings = np.zeros([len(loader.dataset), embedding_size])
    with torch.no_grad():
        for idx, (image, _) in enumerate(
            tqdm(loader, desc="Create embeddings matrix", total=len(loader)),
        ):
            embeddings[idx, :] = F.normalize(backbone(image.to(device))).cpu()

    # faces 에 crop_path, face_idx, face 저장
    i = 0
    faces = []
    for crop_path, _ in dataset.samples:
      face = cv2.imread(crop_path)
      faces.append(
          {
            "crop_path":crop_path, 
            "face_idx" : i,
            "face": face
          }
      )
      i = i + 1

    #  faces에 embeddings 넣기
    for i, embedding in enumerate(embeddings):
      faces[i]["embedding"] = embedding

    return faces, embeddings


def grouping(faces, images, url_dict, cosine_similaritys, cos_similarity_threshold) : 
    """
    cosine_similarity를 이용해 유사한 얼굴끼리 grouping하는 함수

    Args:
      faces (list): [{
                      "crop_path" : crop_path
                      "face_idx" : 전체 얼굴 중에서 몇번째 얼굴인지 (이게 all_faces의 idx랑 동일함)
                      "face" : [얼굴 pixel RGB] 
                    }]

      images (list) : [{
                      "id" : DB에서 이미지 id
                      "url" : S3에서 생성한 url
                      "group_idx" : 얼굴이 없는 사진에 대해서만 -2 들은 상태. //여기를 완성시키는 것이 최종 목표
                    }]

      url_dict (dictionary) : crop img path로 original url을 찾을 수 있게 만든 dict. key = crop image path, value = images idx
      
      cosine_similaritys (2차원 list) : cosine_similarity[i][j]로 접근 가능

    Returns:
      groups (list): [{
                      "original_images_idx_list" : 해당 crop이미지의 원본 url이 담긴 urls idx
                      "crop_path_list" : crop path
                      "face_list" : crop한 얼굴 이미지 list
                      "face2_idx_list" : all_faces의 idx, global_face_idx.
                      "face1_idx_list" : group 이미지의 어떤 이미지랑 유사도가 0.4가 넘어서 들어왔는지. (빈 리스트이면 맨 처음 추가된 이미지인것.)
                      "cosine_similarity_list" : 그 유사도가 얼마였는지 (-1 이면 맨 처음 추가된 이미지인 것)
                    }]

      images (list) : [{
                      "id" : DB에서 이미지 id
                      "url" : S3에서 생성한 url
                      "group_idx" : [1, 4, 9] (해당 사진이 속한 group의 list를 전송), 얼굴이 없으면 [-2]. 얼굴이 있는데 group에 넣기엔 너무 한장일 경우 [-1].
                    }]

      cos_similarity_threshold (float) : cosine similarity 몇이상을 같은 group으로 묶을지를 정하는 threshold
    """
    # face1 : 비교 기준이 되는 이미지
    # face2 : 지금 그룹을 정해주고 싶은 이미지
    groups = []
    for face2_idx, face2 in enumerate(faces) : # 해당 얼굴에 대해
        is_already_in_group = False

        # Case1 : 기존 그룹 탐색
        for group_idx, group in enumerate(groups): # 각 그룹마다
          for i, face1_idx in enumerate(group["face2_idx_list"]) : #그 그룹에 들어있는 얼굴마다
            cosine_similarity = cosine_similaritys[face1_idx][face2_idx]

            if( not is_already_in_group and cosine_similarity > cos_similarity_threshold ) :
              crop_path = face2["crop_path"]
              original_images_idx = url_dict[crop_path]

              # images에 group idx 넣어주기
              if "group_idx" in images[original_images_idx] :
                images[original_images_idx]["group_idx"].append(group_idx)
              else :
                images[original_images_idx]["group_idx"] = [group_idx]

              # groups 갱신
              group["original_images_idx_list"].append(original_images_idx) #dictionary에서 원래 url idx 찾아넣기
              group["crop_path_list"].append(crop_path)
              group["face_list"].append(face2["face"])
              group["face2_idx_list"].append(face2_idx)
              group["face1_idx_list"].append(face1_idx)
              group["cosine_similarity_list"].append(cosine_similarity)

              is_already_in_group = True
            

        # Case2 : 못 넣었으면 새로운 그룹에 추가
        if not is_already_in_group :
          crop_path = face2["crop_path"]
          original_images_idx = url_dict[crop_path]

          # images에 group idx 넣어주기
          if "group_idx" in images[original_images_idx] :
            images[original_images_idx]["group_idx"].append(len(groups))
          else :
            images[original_images_idx]["group_idx"] = [len(groups)]

          # groups 갱신
          groups.append ({
            "original_images_idx_list" :[original_images_idx],
            "crop_path_list" : [crop_path],
            "face_list" : [face2["face"]],
            "face2_idx_list" : [face2_idx], 
            "face1_idx_list" : [[]],
            "cosine_similarity_list" : [-1]
          })

    print(len(groups))

    # group len이 1인건 group index 0 으로 변경
    print("사진 한장뿐인 그룹 목록")
    for group_idx, group in enumerate(groups):
      if len(group["original_images_idx_list"]) == 1 :
        print(group_idx)
        images_idx = group["original_images_idx_list"][0]
        images[images_idx]["group_idx"].remove(group_idx)

        if -1 not in images[images_idx]["group_idx"] :
          images[images_idx]["group_idx"].append(-1)

    return groups, images


#----------------------------------------------------------
#                        main 
#----------------------------------------------------------
def face_recognition(images):
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

      images (list) : [{
                      "id" : DB에서 이미지 id
                      "url" : S3에서 생성한 url
                      "group_idx" : [1, 4, 9] (해당 사진이 속한 group의 list를 전송), 얼굴이 없으면 [-2]. 얼굴이 있는데 group에 넣기엔 너무 한장일 경우 [-1].
                    }]
    """
    #base_folder = "/content/drive/MyDrive/Capstone_Face_Test"
    cos_similarity_threshold = 0.45

    url_dict = {}
    input_size = 112
    #crop_base_folder = base_folder + "/crop_images"
    crop_base_folder = "./crop_images"
    crop_folder = crop_base_folder + "/crop"

    if not os.path.isdir(crop_base_folder):
        os.mkdir(crop_base_folder)

    if os.path.exists(crop_folder):
      shutil.rmtree(crop_folder)
      os.mkdir(crop_folder)
    else :
      os.mkdir(crop_folder)


    #----------------------------------------------------------
    #                      Step1. face detect
    #----------------------------------------------------------
    print("Step1. face detect\n")

    for idx, image in enumerate(images):
      # s3에서 생성된 url을 request&response로 받아서 img로 넘기기
      res = request.urlopen(image["url"]).read()
      img = Image.open(BytesIO(res))
      img = img.convert('RGB')
      pixels = np.asarray(img)  

      # face detect 얼굴 탐지
      images, url_dict = face_crop(idx, pixels, images, crop_folder, url_dict)


    #----------------------------------------------------------
    #                      Step2. embedding
    #----------------------------------------------------------
    print("\n\n")
    print("Step2. embedding\n")

    faces, embeddings = get_embeddings(
            data_root = crop_base_folder ,
            model_root = "./face_recognition/checkpoint/backbone_ir50_ms1m_epoch120.pth",
            input_size = [input_size, input_size],
        )
    print("faces 길이 = ", len(faces))
    print("embeddings 길이 = ", len(embeddings)) 


    #----------------------------------------------------------
    #                      Step3. cos_similarity 계산
    #----------------------------------------------------------
    print("\n\n")
    print("Step3. cos_similarity 계산\n")

    cosine_similaritys = np.dot(embeddings, embeddings.T)
    cosine_similaritys = cosine_similaritys.clip(min=0, max=1)


    #----------------------------------------------------------
    #                      Step4. grouping
    #----------------------------------------------------------
    print("\n\n")
    print("Step4. grouping\n")

    groups, images = grouping(faces, images, url_dict, cosine_similaritys, cos_similarity_threshold)
    
    return groups, images