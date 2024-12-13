# SDV_AIFW
HW-aware SDV AI framework

## 개요

- **SDV 기반 자동차 SW 플랫폼**  
  - 차량 내 제한된 리소스와 신뢰성을 충족하며 고속 AI 처리 지원  
  - OTA 업데이트 및 개인화 기능을 통해 새로운 비즈니스 모델 창출  

- **온디바이스 AI 통합 개발 프레임워크**  
  - HPC AI 가속 및 최적화 개발 도구 제공  
  - 국산 AI 반도체 지원 가능한 SW 플랫폼 구축  

- **필요성 및 기대 효과**  
  - 국내 자동차 산업의 글로벌 경쟁력 강화  
  - SDV 기반 구독 경제 모델로 새로운 비즈니스 기회 창출


<div align="center">
  <img src="https://github.com/user-attachments/assets/3127675f-4a1f-4e8d-b778-94e1f4f3740a" width="500">
</div>

## 프로젝트 목표

- **SDV 차량에 최적화된 온디바이스 및 서버 환경용 AI 프레임워크를 개발하여 개인화 학습 등 기존 프레임워크에 없는 기능을 추가 제공할 예정임**

- **개발자 요구사항을 반영한 IDE 환경 및 편의성 기능을 포함하여 효율적인 개발 지원을 목표로 함**  

- **온디바이스 기종에 따라 AI 가속기를 활용할 수 있는 hardware-aware 최적화 기능을 제공할 계획임**

## 모
<details>
  <summary>Monolithic & Microservice</summary>
  - ****
  - Show what is needed for on device/on premise environment
</details>

## 온디바이스 개발 환경
- ML 지원 AI 프레임워크는 다양한 기능 도구를 활용한 AI 서비스를 제공할 예정임.  
  AI 서비스를 제공하기 위해 온디바이스 환경 내 기능 도구별로 환경 구축 및 실행이 가능한 Docker Container를 활용하여 각 기능 도구별 환경 구축이 필요함.

    <details>
      <summary>System Requirements</summary>
      - 시스템 요구사항은 기능도구들에게 공통으로 요구되며 요구사항은 아래와 같음
      <div align="center">
        <img src="https://github.com/user-attachments/assets/ce1c7508-8ffe-442d-9ecd-28cb53967099" width="500">
      </div>
    </details>
    
    <details>
      <summary>Training Requirements</summary>
      - Training 기능도구는 인공지능 모델을 학습하는 기능도구로써 인공지능 모델 학습에 필요한 요구사항은 아래와 같음
      <div align="center">
        <img src="https://github.com/user-attachments/assets/73713270-5d59-4e6e-af31-47683b6060c5" width="500">
      </div>
    </details>
    
    <details>
      <summary>Inference Requirements</summary>
      - Inference 기능도구는 인공지능 모델을 추론하는 기능도구로써 추론 정보를 AI 서비스에 활용할수있도록 분석 및 시각화 패키지가 요구됨. 인공지능 모델 추론에 필요한 요구사항은 아래와 같음
      <div align="center">
        <img src="https://github.com/user-attachments/assets/b1df4a8e-a227-4872-80d4-183d676bb27e" width="500">
      </div>
    </details>
    
    <details>
      <summary>Optimization Requirements</summary>
      - Optimization 기능도구는 인공지능 모델을 최적화하는 기능도구로써 hardware-aware 하게 온디바이스 요구사항에 맞추어 인공지능을 최적화를 실시함. 인공지능 모델 최적화에 필요한 요구사항은 아래와 같음
      <div align="center">
        <img src="https://github.com/user-attachments/assets/4948ef47-b2ef-43cf-a3e1-2464007a59d5" width="500">
      </div>
    </details>



<details>
  <summary>Docker 설치 방법</summary>
  - kuberentes function explaination
</details>

<details>
  <summary>CUDA Requirements</summary>
  - kuberentes function explaination
</details>

## 설치

  - 실행 명령어 추가
  - requirements.txt 표 추가

## Functions
<details>
  <summary>Training</summary>
  - YOLOv5 is used for object detection and identifying bounding boxes.
</details>

<details>
  <summary>Inference</summary>
  - StrongSORT is used for object tracking across multiple frames.
</details>

<details>
  <summary>Optimization</summary>
  - OSNet is used for object classification and re-identification.
</details>

## Contact
