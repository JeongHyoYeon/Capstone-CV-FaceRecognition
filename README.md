# Face Recognition


### ✅FaceRecognition.ipynb
MTCNN 모델으로 face detect하여 얼굴 부분만 crop해서 crop/group_name 폴더에 저장한 후

crop 폴더를 data load 하여 ArcFace를 이용해 embedding하고

embedding으로 cosine similarity를 계산한 후

이를 이용해 동일인물사진끼리 grouping하는 코드입니다.


- FaceRecognition_Floder.ipynb : input이 images/group_name/user_name 폴더일때 코드

- FaceRecognition_URL.ipynb : Backend 서버에서 AWS s3에 이미지를 저장시키고 그 이미지 url을 dictionary 형태로 input으로 받아 Face Recognition 한 후 grouping 하는 코드



### ✅이용 모델
- MTCNN (face detection)
- Arcface (face embedding)
(사진 첨부하기)



### ✅참고 논문 & 깃허브


