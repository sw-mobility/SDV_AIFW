# Frontend Project Guide

전체 프로젝트의 구조를 설명합니다.
## 컴포넌트 조립법 예시: [FrontendArchitectureGuide.md](FrontendArchitectureGuide.md)

## Project Structure

```
front/
├── src/                           # 소스 코드 메인 폴더
│   ├── main.jsx                   # React 앱 진입점
│   ├── index.css                  # 전역 스타일 및 테마 설정 (이 프로젝트의 테마 색상들과 여백 설정 -> index.css를 참고해서 개별 css 작업)
│   ├── app/                       # 앱 설정 및 라우팅
│   │   ├── App.jsx               # 메인 앱 컴포넌트
│   │   ├── App.css               # 앱 전역 스타일
│   │   └── routes/
│   │       └── AppRoutes.jsx     # 라우팅 설정
│   ├── pages/                     # 페이지 컴포넌트들
│   │   ├── index_page/           # 메인 대시보드
│   │   │   ├── IndexPage.jsx
│   │   │   ├── IndexPage.module.css
│   │   │   └── index_tab/
│   │   │       ├── ProjectsTab.jsx
│   │   │       └── DatasetsTab.jsx
│   │   ├── training_page/        
│   │   │   ├── TrainingPage.jsx
│   │   │   └── TrainingPage.module.css
│   │   ├── labeling_page/       
│   │   │   ├── LabelingPage.jsx
│   │   │   └── LabelingPage.module.css
│   │   ├── optimization_page/  
│   │   │   ├── OptimizationPage.jsx
│   │   │   └── OptimizationPage.module.css
│   │   ├── validation_page/   
│   │   │   ├── ValidationPage.jsx
│   │   │   └── ValidationPage.module.css
│   │   ├── project_home/         
│   │   │   ├── ProjectHomePage.jsx
│   │   │   └── ProjectHomePage.module.css
│   │   ├── DeploymentPage.jsx
│   │   └── ServiceProcessPage.jsx
│   ├── components/                # 재사용 가능한 UI 컴포넌트들 (각 목적은 파일 상단 주석에서 확인 가능)
│   │   ├── ui/                   # 기본 UI 컴포넌트
│   │   │   ├── Button.jsx
│   │   │   ├── Card.jsx
│   │   │   ├── Modal.jsx
│   │   │   ├── Table.jsx
│   │   │   ├── ProgressBar.jsx
│   │   │   ├── CodeEditor.jsx
│   │   │   ├── Selector.jsx
│   │   │   ├── Skeleton.jsx
│   │   │   ├── StatusChip.jsx
│   │   │   ├── TabNavigation.jsx
│   │   │   ├── SectionTitle.jsx
│   │   │   ├── Loading.jsx
│   │   │   ├── ErrorMessage.jsx
│   │   │   ├── EmptyState.jsx
│   │   │   ├── FileUploadField.jsx
│   │   │   ├── ShowMoreGrid.jsx
│   │   │   └── CreateModal.jsx
│   │   ├── layout/               # 레이아웃 컴포넌트
│   │   │   ├── MainLayout.jsx
│   │   │   ├── Header.jsx
│   │   │   ├── Sidebar.jsx
│   │   │   └── Footer.jsx
│   │   ├── common/               # 공통 컴포넌트
│   │   │   └── DeleteConfirmModal.jsx
│   │   └── features/             # 기능별 컴포넌트
│   │       ├── dataset/          # 데이터셋 관련
│   │       │   ├── DatasetDrawer.jsx
│   │       │   ├── DatasetDataPanel.jsx
│   │       │   ├── DatasetEditModal.jsx
│   │       │   ├── DatasetUploadModal.jsx
│   │       │   ├── DatasetUploadFilesModal.jsx
│   │       │   ├── RawDatasetTable.jsx
│   │       │   └── Dataset.module.css
│   │       ├── training/         # 훈련 관련
│   │       │   ├── AlgorithmSelector.jsx
│   │       │   ├── DatasetSelector.jsx
│   │       │   ├── TrainingTypeSelector.jsx
│   │       │   ├── ParameterSelector.jsx
│   │       │   ├── ParameterEditor.jsx
│   │       │   ├── ParameterSection.jsx
│   │       │   ├── TrainingExecution.jsx
│   │       │   ├── ExpertModeToggle.jsx
│   │       │   └── ContinualLearningInfo.jsx
│   │       ├── labeling/         # 라벨링 관련
│   │       │   ├── LabelingWorkspace.jsx
│   │       │   ├── DatasetTablePanel.jsx
│   │       │   └── DatasetTablePanel.module.css
│   │       ├── optimization/     # 최적화 관련
│   │       │   ├── ModelSelector.jsx
│   │       │   ├── TargetBoardSelector.jsx
│   │       │   ├── TestDatasetSelector.jsx
│   │       │   ├── OptionEditor.jsx
│   │       │   └── OptionEditor.module.css
│   │       └── validation/       # 검증 관련
│   │           ├── ValidationWorkspace.jsx
│   │           ├── MetricSelector.jsx
│   │           ├── ResultsTable.jsx
│   │           ├── StatusBadge.jsx
│   │           └── Validation.module.css
│   ├── hooks/                    # 비즈니스 로직 및 상태 관리
│   │   ├── index/               # 메인 페이지 관련 훅
│   │   │   ├── useIndexTabs.js
│   │   │   ├── useProjects.js
│   │   │   └── useDatasets.js
│   │   ├── training/            # 훈련 관련 훅
│   │   │   ├── useTrainingState.js
│   │   │   ├── useTrainingCore.js
│   │   │   ├── useTrainingDatasets.js
│   │   │   ├── useTrainingExecution.js
│   │   │   ├── useTrainingSnapshots.js
│   │   │   └── useTrainingUI.js
│   │   ├── labeling/            # 라벨링 관련 훅
│   │   │   ├── useLabeling.js
│   │   │   └── useLabelingWorkspace.js
│   │   ├── optimization/        # 최적화 관련 훅
│   │   │   └── useOptimizationState.js
│   │   ├── validation/          # 검증 관련 훅
│   │   │   └── useValidation.js
│   │   ├── dataset/             # 데이터셋 관련 훅
│   │   │   ├── useDatasetData.js
│   │   │   └── useDatasetUpload.js
│   │   ├── common/              # 공통 훅
│   │   │   └── useProgress.js
│   │   └── index.js             # 훅 export 파일 (page 나 컴포넌트에서 hook을 불러와 사용할 땐 이 파일만 import하면 되도록)
│   ├── api/                     # 서버 통신 API
│   │   ├── datasets.js          # 데이터셋 API
│   │   ├── projects.js          # 프로젝트 API
│   │   └── uid.js               # 사용자 ID API
│   ├── domain/                  # 비즈니스 규칙 및 상수
│   │   └── training/
│   │       ├── parameterGroups.js
│   │       ├── trainingTypes.js
│   │       └── trainingValidation.js
│   ├── mocks/                   # dummy (아직 api 연동 안 되었지만 필요한 data)
│   │   └── trainingSnapshots.js
│   └── storybook/               # Storybook 스토리 파일들
├── public/                       # 정적 파일들 (이미지는 이곳에)
│   └── logo.png
├── package.json                  # 프로젝트 설정 및 의존성
├── vite.config.js               # Vite 빌드 도구 설정
└── README.md                    # 이 문서
```

## Architecture

이 프로젝트는 **4계층 아키텍처**를 따르며, 각 계층은 명확한 책임을 가지고 있습니다.

```
┌─────────────────────────────────────┐
│           Presentation Layer        │  ← 사용자 인터페이스
│         (Pages, Components)         │
├─────────────────────────────────────┤
│           Business Logic Layer      │  ← 비즈니스 로직
│              (Hooks)                │
├─────────────────────────────────────┤
│           Domain Layer              │  ← 핵심 규칙
│            (Domain)                 │
├─────────────────────────────────────┤
│           Data Access Layer         │  ← 데이터 접근
│              (API)                  │
└─────────────────────────────────────┘
```

### 1. Presentation Layer
**위치**: `src/pages/`, `src/components/`
**역할**: 사용자 인터페이스와 사용자 상호작용
**특징**:
- 화면 표시만 담당
- 비즈니스 로직을 직접 포함하지 않음
- 기능이 없는 UI
- Hooks를 통해 데이터와 로직을 받아옴
- JSX 확장자 파일의 return 값은 사용자 인터페이스에 표출되는 내용입니다.


```javascript
// 예시: TrainingPage.jsx
const TrainingPage = () => {
  const {
    isTraining,
    progress,
    logs,
    handleRunTraining
  } = useTrainingState(); // Hook에서 로직을 받아옴

  return (
    <div>
      <TrainingExecution 
        isTraining={isTraining}
        progress={progress}
        logs={logs}
        onRunTraining={handleRunTraining}
      />
    </div>
  );
};
```

### 2. Business Logic Layer
**위치**: `src/hooks/`
**역할**: 데이터 처리, 상태 관리, 비즈니스 규칙 적용
**특징**:
- 컴포넌트에서 사용할 데이터와 로직을 제공
- API 호출과 데이터 변환을 담당
- 재사용 가능한 로직들을 모아둠
- JS 확장자

```javascript
// 예시: useDatasets.js
export const useDatasets = () => {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchDatasets = async () => {
    setLoading(true);
    try {
      const response = await fetchRawDatasets({ uid });
      setDatasets(response.data || []);
    } catch (error) {
      console.error('Error fetching datasets:', error);
    } finally {
      setLoading(false);
    }
  };

  return { datasets, loading, fetchDatasets };
};
```

### 3. Domain Layer (도메인 계층)
**위치**: `src/domain/`
**역할**: 핵심 비즈니스 규칙과 상수
**특징**:
- 다른 layer에 의존하지 않음
- 검증 규칙, 상수, 타입 정의
- JS 확장자

```javascript
// 예시: trainingValidation.js
export const validateTrainingExecution = (config) => {
  const errors = [];
  
  if (!config.selectedDataset) {
    errors.push({ message: 'Dataset is required' });
  }
  
  if (!config.algorithm) {
    errors.push({ message: 'Algorithm is required' });
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};
```

### 4. Data Access Layer (데이터 접근 계층)
**위치**: `src/api/`
**역할**: 외부 데이터 소스와의 통신
**특징**:
- HTTP 요청을 통한 서버 통신
- 데이터 변환과 에러 처리
- 다른 layer에 의존하지 않음
- JS 확장자

```javascript
// 예시: datasets.js
export const fetchRawDatasets = async ({ uid }) => {
  const response = await fetch(`${BASE_URL}/datasets/raw/list?uid=${uid}`);
  
  if (!response.ok) {
    throw new Error('Failed to fetch datasets');
  }
  
  return response.json();
};
```

## Getting Started

### Development Environment Setup
```bash
# 의존성 설치
npm install

# 개발 서버 실행 (보통 localhost:5173에서 실행됨)
npm run dev

# 빌드 (배포용 파일 생성)
npm run build

# 빌드 결과 미리보기
npm run preview
```


## Code Reading Guide

### Step 1: Entry Points
```
src/main.jsx → src/app/App.jsx → src/app/routes/AppRoutes.jsx
```
- **main.jsx**: 앱이 시작되는 지점, React 앱을 DOM에 마운트
- **App.jsx**: 전체 앱의 뼈대 (전역 스타일, Context, 공통 레이아웃)
- **AppRoutes.jsx**: 페이지 간 이동을 관리 (URL에 따라 어떤 페이지를 보여줄지 결정)

### Step 2: Main Page Structure
```
src/pages/index_page/IndexPage.jsx
├── src/pages/index_page/index_tab/ProjectsTab.jsx
└── src/pages/index_page/index_tab/DatasetsTab.jsx
```
- **IndexPage.jsx**: 메인 페이지 (프로젝트/데이터셋 탭으로 구성)
- **ProjectsTab.jsx**: 프로젝트 목록 관리 (생성, 조회, 삭제 등)
- **DatasetsTab.jsx**: 데이터셋 목록 관리 (업로드, 조회, 삭제 등)

### Step 3: Feature Pages
```
src/pages/
├── training_page/TrainingPage.jsx      # 모델 훈련
├── labeling_page/LabelingPage.jsx      # 데이터 라벨링
├── optimization_page/OptimizationPage.jsx  # 모델 최적화
└── validation_page/ValidationPage.jsx  # 모델 검증
```

### Step 4: Business Logic (Hooks)
```
src/hooks/
├── index/useIndexTabs.js      # 탭 관리 (활성 탭 상태, 탭 전환)
├── index/useProjects.js       # 프로젝트 관리 (CRUD 작업)
├── index/useDatasets.js       # 데이터셋 관리 (업로드, 조회, 삭제)
├── training/useTrainingState.js  # 훈련 상태 관리 (진행률, 로그 등)
```

## Key Directories Explained

### `src/pages/` - Page Components
**역할**: 사용자가 보는 화면들을 담당
**특징**:
- 각 페이지는 하나의 완성된 화면을 구성
- 여러 컴포넌트를 조합해서 페이지를 만듦
- 사용자 인터랙션의 최상위 레벨

**예시**:
- `TrainingPage.jsx`: 모델 훈련 관련 모든 기능이 포함된 페이지
- `LabelingPage.jsx`: 데이터 라벨링 작업을 위한 페이지

### `src/components/` - UI Components
**역할**: 재사용 가능한 작은 UI 부품

**예시**:
- `ui/Button.jsx`: 프로젝트 전체에서 사용하는 공통 버튼
- `features/dataset/DatasetCard.jsx`: 데이터셋 정보를 표시하는 카드 컴포넌트

### `src/hooks/` - Business Logic
**역할**: 데이터 처리와 상태 관리를 담당
**특징**:
- 컴포넌트에서 사용할 데이터와 로직을 제공
- 서버 통신, 데이터 변환, 상태 관리를 담당
- 여러 컴포넌트에서 재사용 가능한 로직들을 모아둠

**구조**:
```
hooks/
├── index/           # 메인 페이지 관련 로직
├── training/        # 훈련 관련 로직
├── labeling/        # 라벨링 관련 로직
├── optimization/    # 최적화 관련 로직
├── validation/      # 검증 관련 로직
└── common/          # 공통 로직 (비동기 처리, 로컬스토리지 등)
```

**예시**:
- `useDatasets.js`: 데이터셋 목록 조회, 생성, 삭제 등의 로직
- `useTrainingState.js`: 훈련 진행 상태, 로그, 결과 관리

### `src/api/` - Server Communication
**역할**: 백엔드 서버와의 데이터 주고받기
**특징**:
- HTTP 요청(GET, POST, PUT, DELETE)을 통해 서버와 통신
- 요청/응답 데이터 형식 처리
- 에러 처리 및 재시도 로직

**예시**:
- `datasets.js`: 데이터셋 관련 모든 API 호출 (목록 조회, 업로드, 삭제 등)
- `training.js`: 훈련 관련 API 호출 (훈련 시작, 상태 조회, 중단 등)

### `src/domain/` - Business Rules
**역할**: 애플리케이션의 핵심 규칙과 상수들
**특징**:
- 데이터 검증 규칙 (필수 필드, 형식 검사 등)
- 상수 값들 (옵션 목록, 설정 기본값 등)
- 비즈니스 로직의 핵심 규칙들

**예시**:
- `training/trainingValidation.js`: 훈련 설정 검증 규칙
- `training/parameterGroups.js`: 훈련 파라미터 그룹 정의

## Core Concepts

### 1. Component
**정의**: 화면의 일부분을 담당하는 재사용 가능한 부품

**특징**: UI 렌더링과 사용자 인터랙션 처리

```javascript
// Button.jsx - 버튼 컴포넌트
function Button({ text, onClick, disabled = false }) {
    return (
        <button 
            onClick={onClick} 
            disabled={disabled}
            className="custom-button"
        >
            {text}
        </button>
    );
}
```

### 2. Hook
**정의**: 데이터 처리와 상태 관리를 담당하는 함수

**특징**: 컴포넌트에서 사용할 데이터와 로직을 제공

```javascript
// useDatasets.js - 데이터셋 관리 훅
function useDatasets() {
    const [datasets, setDatasets] = useState([]);
    const [loading, setLoading] = useState(false);
    
    const fetchDatasets = async () => {
        setLoading(true);
        try {
            const data = await api.getDatasets();
            setDatasets(data);
        } catch (error) {
            console.error('Failed to fetch datasets:', error);
        } finally {
            setLoading(false);
        }
    };
    
    return { datasets, loading, fetchDatasets };
}
```

### 3. Props
**정의**: 부모 컴포넌트에서 자식 컴포넌트로 전달하는 데이터

**특징**: 컴포넌트 간의 데이터 전달 방식

```javascript
// 부모 컴포넌트에서 사용
<Button text="저장하기" onClick={handleSave} disabled={isLoading} />

// 자식 컴포넌트에서 받기
function Button({ text, onClick, disabled }) {
    return <button onClick={onClick} disabled={disabled}>{text}</button>;
}
```

### 4. State
**정의**: 컴포넌트나 훅에서 관리하는 변경 가능한 데이터

**특징**: 상태가 변경되면 관련 컴포넌트가 자동으로 다시 렌더링됨

```javascript
const [isModalOpen, setIsModalOpen] = useState(false);
const [selectedItem, setSelectedItem] = useState(null);
const [datasets, setDatasets] = useState([]);
```

## Data Flow Pattern

### 1. 사용자 액션 → 상태 변경 → UI 업데이트
```
사용자 클릭 → 이벤트 핸들러 → Hook 함수 호출 → API 요청 → 상태 업데이트 → 화면 다시 그리기
```

### 2. 실제 예시: 데이터셋 삭제
```javascript
// 1. 사용자가 삭제 버튼 클릭
<Button onClick={() => handleDelete(dataset.id)} text="삭제" />

// 2. 이벤트 핸들러에서 Hook 함수 호출
const { deleteDataset } = useDatasets();
const handleDelete = (id) => deleteDataset(id);

// 3. Hook에서 API 호출 및 상태 업데이트
const deleteDataset = async (id) => {
    await api.deleteDataset(id);
    setDatasets(prev => prev.filter(item => item.id !== id));
};

// 4. 상태 변경으로 화면 자동 업데이트
```

## Development Guidelines

### Adding New Features

1. **페이지 추가시**
   ```
   src/pages/새기능_page/새기능Page.jsx
   src/pages/새기능_page/새기능Page.module.css
   ```

2. **컴포넌트 추가시**
   ```
   src/components/features/새기능/새기능Component.jsx
   src/components/features/새기능/새기능Component.module.css
   ```

3. **비즈니스 로직 추가시**
   ```
   src/hooks/새기능/use새기능.js
   ```

4. **API 추가시**
   ```
   src/api/새기능.js
   ```

### File Naming Conventions

- **컴포넌트**: PascalCase (예: `DatasetCard.jsx`)
- **훅**: camelCase + use (예: `useDatasets.js`)
- **API**: camelCase (예: `datasets.js`)
- **스타일**: 컴포넌트명 + .module.css (예: `DatasetCard.module.css`)

### CSS Modules 사용법
일반 css와 다르게
CSS Modules는 클래스명이 자동으로 고유한 해시로 변환되기 때문에, 배포 이후 발생하는 전역 충돌을 방지할 수 있습니다. 예를 들어, button이라는 클래스가 다른 컴포넌트의 button과 충돌하지 않습니다.
```javascript
// DatasetCard.module.css
.container {
    padding: 16px;
    border: 1px solid #ddd;
}

.title {
    font-size: 18px;
    font-weight: bold;
}
```

```javascript
// DatasetCard.jsx
import styles from './DatasetCard.module.css';

function DatasetCard({ title }) {
    return (
        <div className={styles.container}>
            <h3 className={styles.title}>{title}</h3>
        </div>
    );
}
```

## Troubleshooting

### 자주 발생하는 문제들

1. **컴포넌트가 렌더링되지 않을 때**
   - 파일 이름과 import/export가 일치하는지 확인
   - 컴포넌트가 올바르게 return 하는지 확인

2. **상태가 업데이트되지 않을 때**
   - setState 함수를 올바르게 사용했는지 확인
   - 배열/객체 상태를 직접 수정하지 않았는지 확인

3. **API 호출이 실패할 때**
   - F12 -> 네트워크 탭에서 요청/응답 확인
   - 백엔드 서버가 실행 중인지 확인

4. **스타일이 적용되지 않을 때**
   - CSS Module 파일을 올바르게 import 했는지 확인
   - 클래스명 오타 확인