# Hooks Structure

이 디렉토리는 React 커스텀 훅들을 관리합니다.

## 구조

```
src/hooks/
├── common/           # 공통 훅들
├── training/         # Training 관련 훅들
├── optimization/     # Optimization 관련 훅들
├── index/           # Index 페이지 관련 훅들
└── index.js         # 모든 훅 export
```

## Common Hooks

### useAsync
비동기 작업의 상태를 관리합니다.

```javascript
import { useAsync } from '../hooks';

const { execute, status, data, error, isLoading } = useAsync(asyncFunction);
```

### useProgress
진행률과 로그를 관리합니다.

```javascript
import { useProgress } from '../hooks';

const { 
  isRunning, 
  progress, 
  status, 
  logs, 
  start, 
  stop, 
  complete, 
  addLog, 
  updateProgress 
} = useProgress();
```

### useLocalStorage
로컬 스토리지와 상태를 동기화합니다.

```javascript
import { useLocalStorage } from '../hooks';

const [value, setValue] = useLocalStorage('key', initialValue);
```

### useDebounce
값의 디바운스를 관리합니다.

```javascript
import { useDebounce } from '../hooks';

const debouncedValue = useDebounce(value, 500);
```

## Training Hooks

### useTrainingState (통합 훅)
모든 training 관련 상태를 관리합니다.

```javascript
import { useTrainingState } from '../hooks';

const {
  // Core state
  trainingType,
  algorithm,
  algoParams,
  
  // Dataset state
  datasets,
  selectedDataset,
  
  // Snapshot state
  snapshots,
  selectedSnapshot,
  
  // Execution state
  isTraining,
  progress,
  logs,
  
  // Event handlers
  handleAlgorithmChange,
  handleRunTraining,
} = useTrainingState();
```

### 개별 Training 훅들
- `useTrainingCore`: 핵심 training 상태
- `useTrainingDatasets`: 데이터셋 관리
- `useTrainingSnapshots`: 스냅샷 관리
- `useTrainingExecution`: 실행 관리
- `useTrainingUI`: UI 상태 관리

## Optimization Hooks

### useOptimizationState
Optimization 관련 상태를 관리합니다.

```javascript
import useOptimizationState from '../hooks/index.js';

const {
  targetBoard,
  model,
  testDataset,
  isRunning,
  progress,
  logs,
  runOptimization,
} = useOptimizationState();
```

## Index Hooks

### useIndexTabs
Index 페이지의 탭 관리를 담당합니다.

```javascript
import { useIndexTabs } from '../hooks';

const {
  activeTab,
  tabs,
  handleTabChange
} = useIndexTabs();
```

### useProjects
프로젝트 관련 모든 로직을 관리합니다.

```javascript
import { useProjects } from '../hooks';

const {
  // 상태
  projects,
  loading,
  error,
  isCreateModalOpen,
  isEditModalOpen,
  editProject,
  
  // 핸들러
  handleCreateProject,
  handleEditProject,
  handleDeleteProject,
  handleProjectClick,
  openCreateModal,
  closeCreateModal,
  openEditModal,
  closeEditModal,
  
  // 유틸리티
  fetchProjectsList
} = useProjects();
```

### useDatasets
데이터셋 관련 모든 로직을 관리합니다.

```javascript
import { useDatasets } from '../hooks';

const {
  // 상태
  dataType,
  loading,
  error,
  initialLoading,
  isCreateModalOpen,
  isEditModalOpen,
  isUploadModalOpen,
  isDataPanelOpen,
  editData,
  uploadTarget,
  dataPanelTarget,
  downloadingId,
  deletingId,
  
  // 핸들러
  handleDownload,
  handleEdit,
  handleDelete,
  handleUpload,
  handleCardClick,
  handleDataTypeChange,
  
  // 모달 핸들러
  openCreateModal,
  closeCreateModal,
  openEditModal,
  closeEditModal,
  openUploadModal,
  closeUploadModal,
  openDataPanel,
  closeDataPanel,
  
  // 유틸리티
  fetchDatasetsList,
  refreshCurrentDatasets,
  getCurrentDatasets
} = useDatasets();
```

## 사용법

### 전체 import
```javascript
import { 
  useTrainingState, 
  useAsync,
  useProgress,
  useIndexTabs,
  useProjects,
  useDatasets
} from '../hooks';
import useOptimizationState from '../hooks/index.js';
```

### 개별 import
```javascript
import { useTrainingState } from '../hooks/training/useTrainingState';
import { useAsync } from '../hooks/common/useAsync';
```
