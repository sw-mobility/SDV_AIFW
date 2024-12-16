# SDV_AIFW


## 1. 개요

- **SDV 기반 자동차 SW 플랫폼**  
  - **차량 내 제한된 리소스와 신뢰성을 충족하며 고속 AI 처리 지원**  
  - **OTA 업데이트 및 개인화 기능을 통해 새로운 비즈니스 모델 창출**  

- **온디바이스 AI 통합 개발 프레임워크**  
  - **HPC AI 가속 및 최적화 개발 도구 제공**  
  - **국산 AI 반도체 지원 가능한 SW 플랫폼 구축**  

- **필요성 및 기대 효과**  
  - **국내 자동차 산업의 글로벌 경쟁력 강화**
  - **SDV 기반 구독 경제 모델로 새로운 비즈니스 기회 창출**


<div align="center">
  <img src="https://github.com/user-attachments/assets/3127675f-4a1f-4e8d-b778-94e1f4f3740a" width="600">
</div>

## 2. 공개SW 서비스 제공

- **AI 가속화 및 최적화를 지원하는 API제공**
- **개인화 학습과 컨테이너 오케스트레이션 기능을 통해 온디바이스와 클라우드 환경 모두에서 효율적 AI 서비스 개발과 운영 지원**
- **표준화된 개발 환경과 국산 AI 반도체 활용을 촉진하며, 커뮤니티 기반 협업과 기여를 활성화하여 지속적인 소프트웨어 개선 도모**

## 3. 프로젝트 목표

- **SDV 차량에 최적화된 온디바이스 및 서버 환경용 AI 프레임워크를 개발하여 개인화 학습 등 기존 프레임워크에 없는 기능을 추가 제공할 예정임**

- **개발자 요구사항을 반영한 IDE 환경 및 편의성 기능을 포함하여 효율적인 개발 지원을 목표로 함**  

- **온디바이스 기종에 따라 AI 가속기를 활용할 수 있는 hardware-aware 최적화 기능을 제공할 계획임**
<div align="center">
  <img src="https://github.com/user-attachments/assets/f5883457-7995-46a8-9dcd-1ce0778c2a0f" width="600">
</div>

## 4. Monolithic & Microservice Architecture

<details>
  <summary>Monolithic & Microservice 요구사항</summary>

### **Monolithic Architecture 요구사항**
- 컴퓨팅 자원이 풍부하지 않은 온디바이스 환경 내 AI 프레임워크 구축을 위해 구조적으로 간단한 Monolithic Architecture가 필요함.
- 온디바이스 환경은 차량 탑승자만 사용하기에 single user에 적합하며, Monolithic Architecture로 온디바이스용 AI 프레임워크를 개발할 예정임.
- Monolithic Architecture 기반 쿠버네티스를 구축하기 위한 요구사항은 아래와 같음:

#### Monolithic Architecture 구성 예시
<div align="center">
  <img src="https://github.com/user-attachments/assets/c478a5ab-4475-4648-b4a6-574a0c749060" width="600">
</div>

#### Monolithic Architecture 시스템 요구사항 표
<div align="center">
  <img src="https://github.com/user-attachments/assets/18590ddf-ecc4-43ef-a4c1-de530589f85b" width="600">
</div>


---

### **Microservice Architecture 요구사항**
- 컴퓨팅 자원이 풍부한 온프레미스 환경 내 AI 프레임워크 구축을 위해 scalability가 크고 기능 도구 관리가 용이한 Microservice Architecture가 필요함.
- Multiple user 지원 및 서비스 운영에 장점이 있는 Microservice Architecture로 온프레미스용 AI 프레임워크를 개발할 예정임.
- Microservice Architecture 기반 쿠버네티스를 구축하기 위한 요구사항은 아래와 같음:

#### Microservice Architecture 구성 예시
<div align="center">
  <img src="https://github.com/user-attachments/assets/46b402ca-fe5e-4353-92d1-25da2a0c2951" width="600">
</div>

#### Microservice Architecture 시스템 요구사항 표
<div align="center">
  <img src="https://github.com/user-attachments/assets/a8e0eedb-7202-4065-8e3e-883f795f0d71" width="600">
</div>

#### 네트워크 요구사항 표
<div align="center">
  <img src="https://github.com/user-attachments/assets/7799b665-0d37-4581-a206-43e427349171" width="600">
</div>

---

### **MA/MSA 요구사항 비교**
<div align="center">
  <img src="https://github.com/user-attachments/assets/4738cd28-7c52-40fc-bd8d-983dd7f56901" width="600">

  
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/1817537d-14da-4444-b92e-731368418847" width="600">
</div>

</details>

<details>
  <summary>쿠버네티스 환경 구축</summary>
  쿠버네티스 환경에서는 중앙 제어(Control Panel)가 API 서버, 스케줄러, 컨트롤 매니저 등을 통해 워커 노드에 있는 Pod와 컨테이너를 관리하여 효율적인 애플리케이션 배포 및 실행을 지원합니다.
  <div align="center">
    <img src="https://github.com/user-attachments/assets/98042625-ddce-4cf4-934d-11d5075e026f" alt="Kubernetes Environment" width="600">
  </div>

  ### MA 쿠버네티스 상세 구조도
  Monolithic Architecture에서는 모든 기능이 하나의 Pod 내부에 통합되어 관리되며, 외부 DB와 연결되어 Training, Inference, Optimization 기능을 단일 컨테이너에서 수행합니다.
  <div align="center">
    <img src="https://github.com/user-attachments/assets/c7af82f1-540c-4b9b-92b6-3b04e54a9b92" alt="MA Kubernetes Detailed Diagram" width="600">
  </div>

  ### MSA 쿠버네티스 상세 구조도
  Monolithic Architecture에서는 모든 기능이 하나의 Pod 내부에 통합되어 관리되며, 외부 DB와 연결되어 Training, Inference, Optimization 기능을 단일 컨테이너에서 수행합니다.
  <div align="center">
    <img src="https://github.com/user-attachments/assets/2c926532-dafc-47ba-af3c-ae328f9ef8b8" alt="MSA Kubernetes Detailed Diagram" width="600">
  </div>
</details>


## 5. 온디바이스 개발 환경
- ML 지원 AI 프레임워크는 다양한 기능 도구를 활용한 AI 서비스를 제공할 예정임.  
  AI 서비스를 제공하기 위해 온디바이스 환경 내 기능 도구별로 환경 구축 및 실행이 가능한 Docker Container를 활용하여 각 기능 도구별 환경 구축이 필요함.

    <details>
      <summary>System Requirements</summary>
      - 시스템 요구사항은 기능도구들에게 공통으로 요구되며 요구사항은 아래와 같음
      <div align="center">
        <img src="https://github.com/user-attachments/assets/ce1c7508-8ffe-442d-9ecd-28cb53967099" width="600">
      </div>
    </details>
    
    <details>
      <summary>Training Requirements</summary>
      - Training 기능도구는 인공지능 모델을 학습하는 기능도구로써 인공지능 모델 학습에 필요한 요구사항은 아래와 같음
      <div align="center">
        <img src="https://github.com/user-attachments/assets/73713270-5d59-4e6e-af31-47683b6060c5" width="600">
      </div>
    </details>
    
    <details>
      <summary>Inference Requirements</summary>
      - Inference 기능도구는 인공지능 모델을 추론하는 기능도구로써 추론 정보를 AI 서비스에 활용할수있도록 분석 및 시각화 패키지가 요구됨. 인공지능 모델 추론에 필요한 요구사항은 아래와 같음
      <div align="center">
        <img src="https://github.com/user-attachments/assets/b1df4a8e-a227-4872-80d4-183d676bb27e" width="600">
      </div>
    </details>
    
    <details>
      <summary>Optimization Requirements</summary>
      - Optimization 기능도구는 인공지능 모델을 최적화하는 기능도구로써 hardware-aware 하게 온디바이스 요구사항에 맞추어 인공지능을 최적화를 실시함. 인공지능 모델 최적화에 필요한 요구사항은 아래와 같음
      <div align="center">
        <img src="https://github.com/user-attachments/assets/4948ef47-b2ef-43cf-a3e1-2464007a59d5" width="600">
      </div>
    </details>

## 6. 도커 이미지 가져오고 명령 실행

![image](https://github.com/user-attachments/assets/c81174a1-4f99-4604-9dd9-3f4b6aa62c47)

---

## (a) Docker 이미지 빌드 및 푸시

**Docker 이미지 빌드**  
Dockerfile을 기반으로 Docker 이미지를 생성합니다:
```bash
docker build -t junh27/yolo_app:latest C:\Users\KETI\Desktop\yolo_again
```
Docker Hub에 이미지 푸시
Docker Hub에 이미지를 업로드합니다:

```bash
  docker push junh27/yolo_app:latest
```

## (b) Minikube 시작

### Minikube 클러스터 시작
Minikube 클러스터를 시작합니다:
  ```bash
  minikube start
```

### Minikube Docker 환경 설정
Minikube의 Docker 환경에서 이미지를 활용할 수 있도록 설정합니다:
  ```bash
  eval $(minikube docker-env)
```
## (c) Deployment 생성
### deployment.yaml 파일 적용
Kubernetes Deployment를 생성합니다:
```bash
  kubectl apply -f C:\Users\KETI\Desktop\yolo_again\deployment.yaml
```

### Deployment 상태 확인
Pod가 제대로 생성되었는지 확인합니다:
```bash
  kubectl get deployments
  kubectl get pods
```
## (d) Service 생성
### service.yaml 파일 적용
Kubernetes Service를 생성합니다:
```bash
kubectl apply -f C:\Users\KETI\Desktop\yolo_again\service.yaml
```
### Service 상태 확인
서비스가 제대로 생성되었는지 확인합니다:
```bash
kubectl get services
```
### 서비스 노드포트 확인
생성된 노드포트를 확인합니다:
```bash
kubectl describe service yolo-app
```
## (e) 애플리케이션 접근
### Minikube 서비스 URL 확인
애플리케이션 URL을 가져옵니다:
```bash
minikube service yolo-app --url
```
브라우저 또는 API 클라이언트로 접속
출력된 URL을 복사하여 브라우저나 API 클라이언트에서 접속합니다.

## (f) 추가 명령어
Pod 로그 확인:
```bash
kubectl logs <pod-name> (Pod 이름은 kubectl get pods 명령어로 확인 가능)
```
Pod 삭제 및 재배포:
```bash
kubectl delete -f C:\Users\KETI\Desktop\yolo_again\deployment.yaml
kubectl apply -f C:\Users\KETI\Desktop\yolo_again\deployment.yaml
```
Service 삭제 및 재생성:
```bash
kubectl delete -f C:\Users\KETI\Desktop\yolo_again\service.yaml
kubectl apply -f C:\Users\KETI\Desktop\yolo_again\service.yaml
```

![image](https://github.com/user-attachments/assets/9951b3c7-2cd4-454b-80e6-f06fc43a74fa)

![Screenshot from 2024-10-10 09-37-23](https://github.com/user-attachments/assets/17a1d9b9-84c0-4653-8ff3-4aff7f2196a6)

## 7. Contact
한국전자기술연구원 모빌리티 플랫폼연구센터
- **장수현 (Soohyun Jang)** 책임연구원 / [shjang@keti.re.kr](mailto:shjang@keti.re.kr)
- **장준혁 (JunHyuk Jang)** 선임연구원 / [junjang9327@keti.re.kr](mailto:junjang9327@keti.re.kr)
## 8. Acknowledgement
This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIP) (No.RS-2024-00397615, Development of an automotive software platform for Software-Defined-Vehicle (SDV) integrated with an AI framework required for intelligent vehicles)
