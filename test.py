# user에게 받은 사진을 AWS S3에 저장하고 그 url을 input으로 받는다.
images = [
    {
        "id": 1,
        "url": "https://capstone-aftertrip-test.s3.ap-northeast-2.amazonaws.com/6854d9b5-84a1-4532-b312-98429d9d7671"
    },
    {
        "id": 2,
        "url": "https://capstone-aftertrip-test.s3.ap-northeast-2.amazonaws.com/140b4477-187a-44e8-aed2-2424fb05ee58"
    },
    {
        "id": 3,
        "url": "https://capstone-aftertrip-test.s3.ap-northeast-2.amazonaws.com/41a7e4b4-8130-4f55-a7b1-7950e3b15506"
    },
    {
        "id": 4,
        "url": "https://capstone-aftertrip-test.s3.ap-northeast-2.amazonaws.com/e49e2990-077b-479c-9b00-e69b59c01b00"
    },
    {
        "id": 5,
        "url": "https://capstone-aftertrip-test.s3.ap-northeast-2.amazonaws.com/1d4318f0-3ea2-4ec7-b50f-6e91de1c09d0"
    },
    {
        "id": 6,
        "url": "https://capstone-aftertrip-test.s3.ap-northeast-2.amazonaws.com/7d4b0d96-3121-4ab2-9785-731ecc639f59"
    },
    {
        "id": 7,
        "url": "https://capstone-aftertrip-test.s3.ap-northeast-2.amazonaws.com/b2efe75e-d38b-48ed-ba01-ff54cb3943d7"
    },
    {
        "id": 8,
        "url": "https://capstone-aftertrip-test.s3.ap-northeast-2.amazonaws.com/8941d2e9-4e18-431d-841e-8c78a5b168d0"
    },
    {
        "id": 9,
        "url": "https://capstone-aftertrip-test.s3.ap-northeast-2.amazonaws.com/aa2d53aa-12b2-4d3d-a339-443b08814649"
    },
    {
        "id": 10,
        "url": "https://capstone-aftertrip-test.s3.ap-northeast-2.amazonaws.com/51ae5606-fb57-4d08-a71b-4f2481348471"
    },
    {
        "id": 11,
        "url": "https://capstone-aftertrip-test.s3.ap-northeast-2.amazonaws.com/c6c32076-47b3-4f96-a1f1-ed28c36560b0"
    },
    {
        "id": 12,
        "url": "https://capstone-aftertrip-test.s3.ap-northeast-2.amazonaws.com/1fadcb41-e2ee-467f-9162-f554f22a849d"
    },
    {
        "id": 13,
        "url": "https://capstone-aftertrip-test.s3.ap-northeast-2.amazonaws.com/9ea78401-31d8-4ebe-8b3a-58632fb8d439"
    }
]

# run_face_recog 호출
from face_recognition.main import run_face_recog

groups, group_idx_list, images = run_face_recog(images)