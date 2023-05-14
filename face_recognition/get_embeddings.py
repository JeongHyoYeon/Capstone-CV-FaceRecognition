import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms

from backbone import Backbone

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
    backbone.load_state_dict(torch.load(model_root, map_location=torch.device(device)))
    backbone.to(device)
    backbone.eval()

    # crop 폴더에 들은 얼굴마다 embedding 하기
    embeddings = np.zeros([len(loader.dataset), embedding_size])
    with torch.no_grad():
        for idx, (image, _) in enumerate(
                tqdm(loader, desc="Create embeddings matrix", total=len(loader)),
        ):
            #embeddings[idx, :] = F.normalize(backbone(image.to(device))).cpu()
            embeddings[idx, :] = F.normalize(backbone(image.to(device)))

    # faces 에 crop_path, face_idx, face 저장
    i = 0
    faces = []
    for crop_path, _ in dataset.samples:
        face = cv2.imread(crop_path)
        faces.append(
            {
                "crop_path": crop_path,
                "face_idx": i,
                "face": face
            }
        )
        i = i + 1

    #  faces에 embeddings 넣기
    for i, embedding in enumerate(embeddings):
        faces[i]["embedding"] = embedding

    return faces, embeddings