import os
import cv2

def face_grouping(faces, images, cosine_similaritys, cos_similarity_threshold):
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

      cosine_similaritys (2차원 list) : cosine_similarity[i][j]로 접근 가능

      cos_similarity_threshold (float) : cosine similarity 몇이상을 같은 group으로 묶을지를 정하는 threshold

    Returns:
      groups (list of dictionary): [{
                      "original_images_idx_list" : 해당 crop이미지의 원본이미지 idx
                      "crop_path_list" : crop이미지가 저장된 경로들의 list
                      "face_list" : crop이미지의 list
                      "face2_idx_list" : crop이미지 idx의 list
                      "face1_idx_list" : 해당 crop이미지가 해당 group에 추가될때 group내의 어떤 얼굴과 유사하다 생각되어 들어갔는지. 
                                          그 crop얼굴이미지의 idx의 list. 
                                         (빈 리스트이면 해당 그룹에 맨 처음 추가된 이미지인것)
                      "cosine_similarity_list" : 그때 face2와 face1의 유사도가 얼마였는지. 그 유사도의 list. 
                                                 (-1 이면 맨 처음 추가된 이미지인 것)
                    }]

      group_idx_list (list of int) : 유효한 group의 idx만 들어있는 list. (한 group에 하나의 얼굴만 들어간 경우 해당 그룹은 유효하지 않다 판단)
                                     (group idx 중 
                                      -2 (얼굴이 없는 사진들이 들어가는 그룹), 
                                      -1 (얼굴이 있는데 group에 넣기엔 너무 한장인 사진들이 들어가는 그룹)은 넣지 않았다)

      images (list of dictionary) : [{
                      "id" : DB에서 이미지 id
                      "url" : S3에서 생성한 url
                      "group_idx" : [1, 4, 9] (해당 사진이 속한 group의 list를 전송), 얼굴이 없으면 [-2]. 얼굴이 있는데 group에 넣기엔 너무 한장일 경우 [-1].
                    }])

    """
    # face1 : 비교 기준이 되는 이미지
    # face2 : 지금 그룹을 정해주고 싶은 이미지
    groups = []
    group_idx_list = []
    for face2_idx, face2 in enumerate(faces):  # 해당 얼굴에 대해
        is_already_in_group = False

        # Case1 : 기존 그룹 탐색
        for group_idx, group in enumerate(groups):  # 각 그룹마다
            for i, face1_idx in enumerate(group["face2_idx_list"]):  # 그 그룹에 들어있는 얼굴마다
                cosine_similarity = cosine_similaritys[face1_idx][face2_idx]

                if (not is_already_in_group and cosine_similarity > cos_similarity_threshold):
                    crop_path = face2["crop_path"]
                    original_images_filename = os.path.split(crop_path)[1]  # 전체 경로에서 file name만 parsing
                    original_images_filename = original_images_filename.split('.')[0]  # 1_3.jpg에서 "1_3" parsing
                    original_images_idx = int(original_images_filename.split('_')[0])  # 1_3에서 "1" parsing

                    # images에 group idx 넣어주기
                    if group_idx not in group_idx_list:
                        group_idx_list.append(group_idx)

                    if "group_idx" in images[original_images_idx]:
                        images[original_images_idx]["group_idx"].append(group_idx)
                    else:
                        images[original_images_idx]["group_idx"] = [group_idx]

                    # groups 갱신
                    group["original_images_idx_list"].append(original_images_idx)  
                    group["crop_path_list"].append(crop_path)
                    group["face_list"].append(face2["face"])
                    group["face2_idx_list"].append(face2_idx)
                    group["face1_idx_list"].append(face1_idx)
                    group["cosine_similarity_list"].append(cosine_similarity)

                    is_already_in_group = True

        # Case2 : 못 넣었으면 새로운 그룹에 추가
        if not is_already_in_group:
            crop_path = face2["crop_path"]
            original_images_filename = os.path.split(crop_path)[1]  # 전체 경로에서 file name만 parsing
            original_images_filename = original_images_filename.split('.')[0]  # 1_3.jpg에서 "1_3" parsing
            original_images_idx = int(original_images_filename.split('_')[0])  # 1_3에서 "1" parsing

            # images에 group idx 넣어주기
            if len(groups) not in group_idx_list:
                group_idx_list.append(len(groups))

            if "group_idx" in images[original_images_idx]:
                images[original_images_idx]["group_idx"].append(len(groups))
            else:
                images[original_images_idx]["group_idx"] = [len(groups)]

            # groups 갱신
            groups.append({
                "original_images_idx_list": [original_images_idx],
                "crop_path_list": [crop_path],
                "face_list": [face2["face"]],
                "face2_idx_list": [face2_idx],
                "face1_idx_list": [[]],
                "cosine_similarity_list": [-1]
            })

    print(len(groups))

    # group len이 1인건 group index -1 으로 변경
    #print("사진 한장뿐인 그룹 목록")
    for group_idx, group in enumerate(groups):
        if len(group["original_images_idx_list"]) == 1:
            #print(group_idx)
            images_idx = group["original_images_idx_list"][0]
            images[images_idx]["group_idx"].remove(group_idx)
            group_idx_list.remove(group_idx)

            if -1 not in images[images_idx]["group_idx"]:
                images[images_idx]["group_idx"].append(-1)

    return groups, group_idx_list, images
