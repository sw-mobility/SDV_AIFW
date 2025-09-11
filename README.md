# Folders Explanation (children of api folder)

## Importation hierarchy by folders (Top to bottom)
    core, models, utils, codebase (Used to form service functions)
    |
    services (Used to form routes)
    |
    routes (Defines actual routes)
    |
    main (Aggregates routes to serve)

- .py files in the folders are primarily sorted by its functional difference, such as:
    - dataset
    - project
    - etc.

- They are then sorted again by its kind, such as:
    - raw
    - labeled
    - common
    - etc.

## Detailed Explanation by files

### codebase
    - Contains default codebases to be loaded when system starts.
    - Current codebase files are dummies.
    - Codebases will be 3 or more depending on function. Needs to be defined.
    - Those files can't be changed.

### core
- *config.py*
    - Contains..
        1. Core parameter sets to initialize DB containers
        2. Other settings such as MIME types, API_WORKDIR, etc. to avoid hardcoding fatal params.

- *minio.py*
    - Contains..
        1. MinIO Client which is capable of initialization and all basic I/O tasks
        - *The client is usually called in service functions*

- *mongodb.py*
    - Contains..
        1. MongoDB Client which is capable of initialization and all basic I/O tasks
        - *The client is usually called in service functions*

## models
### *dataset*
- *common_model.py*
    - Contains..
- *labeled_model.py*
    - Contains..
- *raw_model.py*
    - Contains..
















# Problems

### 1. UID 인증 문제
    최초 접속시 라우트에 박치기해서 uid를 전달하는 방식은 프로덕션 빌드에서는 절대 사용할 수 없음
    - UID는 기기인증 도입 시 숨겨질 것
    - UID 필드 외에 uid가 노출되는 경우만 모두 없애면 당분간을 괜찮을 것이라고 판단됨

### 2. 라우트 UID 인증 문제 (완료)
    모든 동작은 uid를 입력받아 사용자 검증, db 초기화 후 진행해야 함
    - 모든 라우트에 init 함수 추가함, init 함수는 users 컬렉션 조회 후 요청 uid랑 매칭, 매칭되지 않을 시 에러 반환

### 3. MIME type 문제 (완료)
    MIME filter 적용 시작했고 전부 적용해야 함 
    - 적용함

### 4. Default assets 문제 (완료)
    minio 공식 이미지를 사용하면 minio가 codebase를 지닌 상태로 활성화될 수 없음
    default assets의 관리를 위해 mongodb도 그것들의 정보를 보유하고 있어야 함
    두 개의 db 모두 커스텀 빌드가 필요할 수 있음
    - api server에 넣어두고 실행 시 minio에 업로드하는 방식으로 진행함
    - 개선이 필요할 수 있음

### 5. Service, Core 코드 비중 문제 (완료하긴 했는데 잘 모르겠음)
    Service에서 처리해야 할 분기가 core에 들어가 있는 경우가 과도하게 많아 core 함수들의 분기를 분해해 각 service에 넣어야 함
    - core 함수들이 더 많아짐
    - 같은 기능이라도 (다운로드, 리스트업 등) 목적에 따라 차별화하려 한 결과물이나 이제 나도 잘 모르겠음

### 6. YOLO - data.yaml 페어링 문제 (완료)
    YOLO의 data.yaml은 training에 필요한 class, data path 정보를 보유
    class 정보는 re-training시 사용자의 의지에 따라 변경돼야 할 수 있지만,
    단순한 파싱으로 이를 커버하려고 할 경우 YOLO의 아키텍처와 맞지 않는 class 정보가 발생할 것
    따라서 data.yaml은 데이터셋과 페어링하고 수정 불가능한 것으로 진행함

### 7. Ultralytics 전체를 임포트해서 사용하는 문제
    model.train() 을 사용해서 training하게 되면 사용자가 커스텀 코드베이스를 사용할 때 코드베이스 전체적인 임포트 경로 문제가 발생할 수 있음
    YOLO에서 training을 할 때 사용되는 파일 및 파이프라인은 고정돼 있음
    따라서 ultralytics 폴더 안의 train.py 파일을 실행시키는 .sh 파일 등을 만드는 것이 더 robust한 방법이라고 판단됨


minio 삭제 메소드를 path 기반으로 전부 교체?

수정:
put raw 만 아직 did 파라미터 수신(완료)
upload는  raw 만 아직 did 송신(완료)
GET /datasets/raw/{did} 는 data_list가 비어 있는 경우 404 처리(완료)
- {did}는 모두 single로 대체됨

0813 수정사항
기존 dataset management에서 dataset primary key를 사용하던 라우트들을 전부 did를 사용하도록 변경함
deletion route들도 작업중

0814 수정사항
list-up 시 uid가 노출되면 안됨. 수정중
근데 did까지도 괜찮은건가? 검토필요