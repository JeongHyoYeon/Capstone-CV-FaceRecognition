# Face Recognition


### ✅FaceRecognition.ipynb
MTCNN 모델으로 face detect하여 얼굴 부분만 crop해서 crop/group_name 폴더에 저장한 후

crop 폴더를 data load 하여 ArcFace를 이용해 embedding하고

embedding으로 cosine similarity를 계산한 후

이를 이용해 동일인물사진끼리 grouping하는 코드입니다.


- <h4>[ver1] FaceRecognition_Floder.ipynb</h4> : input이 images/group_name/user_name 폴더일때 코드

- <h4>[ver2] FaceRecognition_URL.ipynb</h4> : Backend 서버에서 AWS s3에 이미지를 저장시키고 그 이미지 url을 dictionary 형태로 input으로 받아 Face Recognition 한 후 grouping 하는 코드

- <h4>[최종 ver] face_recognition.py </h4> : 리펙토링, 모듈화 마친 코드

### ✅face_recognition.py 실행방법
폴더구조를 아래처럼 두고
```
api
ㄴ함수_호출할_파일.py
ㄴface_recognition
  ㄴface_recognition.py  
  ㄴbackbone.py
  ㄴcheckpoint
    ㄴbackbone_ir50_ms1m_epoch120.pth
```

함수_호출할_파일.py에서 아래처럼 함수 호출로 실행.
```
from face_recognition.face_recognition import face_recognition

groups, images = face_recognition(images)
```


### ✅이용 모델
- MTCNN (face detection) [MTCNN github](https://github.com/ipazc/mtcnn)

- Arcface (face embedding) [Arcface github](https://github.com/spmallick/learnopencv/tree/master/Face-Recognition-with-ArcFace)

<img src="https://user-images.githubusercontent.com/90602936/233304516-94b137ba-91f3-42ec-bb92-5a07e371ae7a.png" width="60%" height="60%">


### ✅결과 시각화
<h3>1. T-SNE를 이용해 embedding 결과의 분포 확인</h3>

- ex1

<img src="https://user-images.githubusercontent.com/90602936/233306231-4ea1569d-e5d5-4668-89fc-d3b446bb2d3b.png" width="50%" height="50%">

- ex2

<img src="https://user-images.githubusercontent.com/90602936/233306337-ea1ce1f1-e4e8-4447-91a4-741e846311b8.png" width="40%" height="40%">

<h3>2. grouping 결과 시각화</h3>

![image](https://user-images.githubusercontent.com/90602936/233305692-e09dcff7-d0aa-4fcb-a045-8a705665c8ba.png)

![image](https://user-images.githubusercontent.com/90602936/233305760-bdeb9993-db15-4ac6-a379-252d4865d972.png)

![image](https://user-images.githubusercontent.com/90602936/233305817-1a66786c-10c8-42b1-93a0-bea51d3fdaae.png)



### ✅ Data Structure
아래 내용과 동일한  .ipynb 파일에 docstring으로 정리되어있습니다.
(노션에 있는거 옮기기)

### ✅ To Do
- 옆면, 얼굴 각도 심하게 돌아간 것에 대해서는 잘 안됨 -> face alignment 추가
- grouping algorithm 개선

### ✅참고 논문 & 깃허브
[github](https://github.com/vinotharjun/FaceGrouping)

