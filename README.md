# Math Word Problem - Korean
자연어 처리 기술을 기반으로 한 [한글 서술형 수학문제 풀이] 공개 저장소입니다.

## 1. Installation
   
    # conda 환경 설치
    conda create -n "mwp-korean" python=3.8.3   
    
    # 설치한 conda 환경 실행
    conda activate mwp-korean
    
    # 현재 GitHub 복사
    git clone https://github.com/jkc-ai/mwp-korean.git
    
    # 실행에 필요한 패키지 설치
    pip install -r requirements.txt
    
## 2. Classifier
   
   * [한글 서술 수학 문제 데이터셋 저장소](https://github.com/jkc-ai/mwp-korean-data)에 공개된 수학 문제 데이터를 분류하는 NLP 분류 모델입니다. 모델의 Encoder는 [KoELECTRA](https://github.com/monologg/KoELECTRA)를 이용하였습니다. 사용법은 다음과 같습니다.

        ```
        python ./classifier/mwp_classifier.py
        ```

   * 위 코드 실행 시, 약 40개 샘플 문제에 대한 모델의 추정 카테고리, 정답 카테고리 및 정답률을 확인하실 수 있습니다.
   * 훈련 코드 및 pre-trained weight는 제공하지 않습니다.

## 3. 지원
이 연구개발은 2021년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행한 연구 성과물의 일부입니다.
해당 연구과제에 대한 정보는 아래와 같습니다.
- 과제번호: 2021-0-02152
- 연구사업명: 인공지능산업원천기술개발
- 연구과제명: 서술형 수학문제 해결 인공지능 알고리즘 개발
- 주관연구기관: 한국원자력연구원
- 공동연구기관: 주식회사 젠티

## 4. 참고자료
[1] KoELECTRA GitHub. https://github.com/monologg/KoELECTRA.

[2] Clark, Kevin and Luong, Minh-Thang and Le, Quoc V and Manning, Christopher D. Electra: Pre-training text encoders as discriminators rather than generators. arXiv preprint arXiv:2003.10555. 2020.

