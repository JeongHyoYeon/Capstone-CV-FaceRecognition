<div align="center">
  <h1>
    Face Recognition
  </h1>
</div>


<img src="https://github.com/JeongHyoYeon/Capstone-CV-FaceRecognition/assets/90602936/93ce7cb5-2b96-432c-b810-96967ae5351f" width="45%" height="45%">
<p>
  첫번째 단계인 FaceDetection은 MTCNN모델으로 하나의 이미지에서 얼굴에 해당하는 모든 부분을 찾아 112x112 사이즈로 crop하였습니다.
  두번째 단계인 FaceEmbedding 은 Arcface 손실함수를 이용한 Backbone모델로 얼굴이미지를 512 사이즈의 embedding vector로 변환하였습니다.
</p>


<img src="https://github.com/JeongHyoYeon/Capstone-CV-FaceRecognition/assets/90602936/f249e4de-d41f-4b06-8c65-2d39e1444773" width="45%" height="45%">
<p>이렇게 embedding된 얼굴들을 t-SNE를 이용해 시각화 해보면 동일인물의 얼굴끼리 가까운 공간에 분포한다는 것을 확인할 수 있습니다.</p>

<img src="https://github.com/JeongHyoYeon/Capstone-CV-FaceRecognition/assets/90602936/b94c465b-15eb-41ac-b0c9-3847433bad04" width="45%" height="45%">
<p> 세번째 단계에서는 모든 얼굴의 embedding vector가 연결된 embeddings vector와 그것의 transpose vector를 내적하여 cosine similarity를 계산하였습니다. </p>
<p> 네번째 단계에서는 crop된 얼굴이미지가 thresholod 값 0.55을 기준으로 유사하다면, original이미지들끼리 grouping하였습니다.</p>

<img src="https://github.com/JeongHyoYeon/Capstone-CV-FaceRecognition/assets/90602936/95cc7b4f-31b3-4b9c-83ce-3c9eeba67ffb" width="45%" height="45%">
<p>최종 결과를 보면 이렇게 동일인물의 사진끼리 하나의 폴더에 잘 들어가있는 것을 확인할 수 있습니다. </p>
<p>이러한 얼굴인식AI는 얼굴 60장에 약 1분이 소요됩니다. </p>

<br>
<div align="center">
  <h1>
    ipynb notebooks
  </h1>
</div>
<h4>[ver1] FaceRecognition_Floder.ipynb</h4> 
: [input] 이미지가 들은 폴더.

<h4>[ver2] FaceRecognition_URL.ipynb</h4> 
: [input] AWS s3에 사용자 이미지를 저장. 해당 이미지들의 url이 저장된 dictionary.

<h4>[최종 ver] face_recognition.py </h4> 
: 리펙토링, 모듈화 마친 코드

<br>
<div align="center">
  <h1>
    실행방법
  </h1>
</div>

```
from main import *

groups, group_idx_list, images = run_face_recog(images)
```

```
Args:
      images (list) : [{
                      "id" : DB에서 이미지 id
                      "url" : S3에서 생성한 url
                    }]
Returns:
      groups (list): [{
                      "original_images_idx_list" : 해당 crop이미지의 원본 url이 담긴 urls idx
                      "crop_path_list" : crop path
                      "face_list" : crop한 얼굴 이미지 list
                      "face2_idx_list" : all_faces의 idx, global_face_idx.
                      "face1_idx_list" : group 이미지의 어떤 이미지랑 유사도가 threshold가 넘어서 들어왔는지. (빈 리스트이면 맨 처음 추가된 이미지인것.)
                      "cosine_similarity_list" : 그 유사도가 얼마였는지 (-1 이면 맨 처음 추가된 이미지인 것)
                    }]

      group_idx_list (list) : 유효한 group의 idx만 들어있는 list. (-2, -1은 넣지 않았다)

      images (list) : [{
                      "id" : DB에서 이미지 id
                      "url" : S3에서 생성한 url
                      "group_idx" : [1, 4, 9] (해당 사진이 속한 group의 list를 전송), 얼굴이 없으면 [-2]. 얼굴이 있는데 group에 넣기엔 너무 한장일 경우 [-1].
                    }]
```

<br>
<div align="center">
  <h1>
    이용 모델
  </h1>
</div>
<h4> MTCNN (face detection) </h4> [MTCNN github](https://github.com/ipazc/mtcnn)
<h4> Arcface (face embedding) </h4> [Arcface github](https://github.com/spmallick/learnopencv/tree/master/Face-Recognition-with-ArcFace)

<br>
<div align="center">
  <h1>
    결과 시각화
  </h1>
</div>
<h3>1. T-SNE를 이용해 embedding 결과의 분포 확인</h3>

- ex1

<img src="https://user-images.githubusercontent.com/90602936/233306231-4ea1569d-e5d5-4668-89fc-d3b446bb2d3b.png" width="50%" height="50%">

- ex2

<img src="https://user-images.githubusercontent.com/90602936/233306337-ea1ce1f1-e4e8-4447-91a4-741e846311b8.png" width="40%" height="40%">

<h3>2. grouping 결과 시각화</h3>

![image](https://user-images.githubusercontent.com/90602936/233305692-e09dcff7-d0aa-4fcb-a045-8a705665c8ba.png)

![image](https://user-images.githubusercontent.com/90602936/233305760-bdeb9993-db15-4ac6-a379-252d4865d972.png)

![image](https://user-images.githubusercontent.com/90602936/233305817-1a66786c-10c8-42b1-93a0-bea51d3fdaae.png)


<br>
<div align="center">
  <h1>
    To Do
  </h1>
</div>
face alignment 추가
grouping algorithm 개선



<div align="center">
  <h1>
    참고자료
  </h1>
</div>
[github](https://github.com/vinotharjun/FaceGrouping)

