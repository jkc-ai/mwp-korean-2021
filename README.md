# mwp-korean
한국어 수학문제 풀이를 위한 공개저장소 입니다.

1. Installation
   
    ```
    conda create -n "mwp-korean" python=3.8.3
    conda activate mwp-korean
    git clone https://github.com/jkc-ai/mwp-korean.git
    pip install -r requirements.txt
    ```

2. Classifier
   
   [한글 서술 수학 문제 데이터셋 저장소](https://github.com/jkc-ai/mwp-korean-data)에 공개된 수학 문제 데이터를 분류하는 NLP 분류 모델입니다. 모델의 Encoder는 [KoELECTRA](https://github.com/monologg/KoELECTRA)를 이용하였습니다. 사용법은 다음과 같습니다.
   
   ```
   python mwp_classifier.py
   ```

   위 코드 실행 시, 샘플 약 40문제에 대한 모델의 추정 카테고리, 정답 카테고리 및 정답률을 확인하실 수 있습니다.
