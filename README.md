<div align="center">
  <h1>
    Project
  </h1>
</div>

<div>
  <p>
    <b>GitHub</b> : <a href="https://github.com/JeongHyoYeon">AfterTripGithub</a>
  </p>
  
  <p>
    <b>AfterTrip</b> : 여행 후 그룹별 사진 공유의 불편함을 해결하기 위해 얼굴인식AI와 객체인식AI를 이용해 폴더를 분류하는 사진 공유용 모바일 웹
  </p>
</div>

<br>
<div align="center">
  <h1>
    Face Recognition
  </h1>
</div>

<div align="center">
<img src="https://github.com/JeongHyoYeon/Capstone-CV-FaceRecognition/assets/90602936/93ce7cb5-2b96-432c-b810-96967ae5351f" width="50%" height="50%">
<p> 
  <b>1. FaceDetection</b>
  : MTCNN모델으로 하나의 이미지에서 얼굴에 해당하는 모든 부분을 찾아 112x112 사이즈로 crop
</p>
<p> 
  <b>2. FaceEmbedding</b>
  : Arcface 손실함수를 이용한 Backbone모델로 얼굴이미지를 512 사이즈의 embedding vector로 변환
</p>


<img src="https://github.com/JeongHyoYeon/Capstone-CV-FaceRecognition/assets/90602936/f249e4de-d41f-4b06-8c65-2d39e1444773" width="50%" height="50%">
<p>
  <b>t-sne로 시각화</b>
  : embedding된 얼굴들을 t-SNE를 이용해 시각화 해보면 동일인물의 얼굴끼리 가까운 공간에 분포
</p>

<img src="https://github.com/JeongHyoYeon/Capstone-CV-FaceRecognition/assets/90602936/b94c465b-15eb-41ac-b0c9-3847433bad04" width="50%" height="50%">
<p> 
  <b>3. Similarity 계산</b>
  : 모든 얼굴의 embedding vector가 연결된 embeddings vector와 그것의 transpose vector를 내적하여 cosine similarity를 계산 
</p>
<p> 
  <b>4. Grouping</b>
  : crop된 얼굴이미지가 thresholod 값 0.55을 기준으로 유사하다면, original이미지들끼리 grouping
</p>

<img src="https://github.com/JeongHyoYeon/Capstone-CV-FaceRecognition/assets/90602936/95cc7b4f-31b3-4b9c-83ce-3c9eeba67ffb" width="50%" height="50%">
<p>
  <b>최종 결과</b>
  : 동일인물의 사진끼리 하나의 폴더
</p>
<p>
  <b>성능</b>
  : 얼굴인식AI는 얼굴 60장에 약 1분 소요 (Tesla T4 GPU 이용)
</p>
</div>

<br>
<div align="center">
  <h1>
    실행 방법
  </h1>
</div>

```
from main import *

groups, group_idx_list, images = run_face_recog(images)
```

```
Args:
      images (list of dictionary) : [{
                      "id" : DB에서 이미지 id
                      "url" : S3에서 생성한 url
                    }]
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
                    }]
```

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
    참고 자료
  </h1>
</div>
<p> <b>MTCNN (face detection)</b> : <a href="https://github.com/ipazc/mtcnn">github</a> </p>
<p> <b>Arcface (face embedding)</b> :  <a href="https://github.com/spmallick/learnopencv/tree/master/Face-Recognition-with-ArcFace">github</a> </p>


<br>
<div align="center">
  <h1>
    결과 시각화
  </h1>
</div>
<h4>1. T-SNE를 이용해 embedding 결과의 분포 확인</h4>

- ex1

<img src="https://user-images.githubusercontent.com/90602936/233306231-4ea1569d-e5d5-4668-89fc-d3b446bb2d3b.png" width="50%" height="50%">

- ex2

<img src="https://user-images.githubusercontent.com/90602936/233306337-ea1ce1f1-e4e8-4447-91a4-741e846311b8.png" width="40%" height="40%">

<h4>2. grouping 결과 시각화</h4>

![image](https://user-images.githubusercontent.com/90602936/233305692-e09dcff7-d0aa-4fcb-a045-8a705665c8ba.png)

![image](https://user-images.githubusercontent.com/90602936/233305760-bdeb9993-db15-4ac6-a379-252d4865d972.png)

![image](https://user-images.githubusercontent.com/90602936/233305817-1a66786c-10c8-42b1-93a0-bea51d3fdaae.png)



