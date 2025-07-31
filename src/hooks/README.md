# Hooks Structure

이 디렉토리는 React 커스텀 훅들을 체계적으로 관리합니다.

## 구조

```
src/hooks/
├── common/           # 공통 훅들
├── training/         # Training 관련 훅들
├── optimization/     # Optimization 관련 훅들
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
import useOptimizationState from '../hooks/useOptimizationState';

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

## 사용법

### 전체 import
```javascript
import { 
  useTrainingState, 
  useAsync,
  useProgress 
} from '../hooks';
import useOptimizationState from '../hooks/useOptimizationState';
```

### 개별 import
```javascript
import { useTrainingState } from '../hooks/training/useTrainingState';
import { useAsync } from '../hooks/common/useAsync';
```

## 장점

1. **단일 책임 원칙**: 각 훅이 하나의 명확한 책임만 가짐
2. **재사용성**: 공통 훅들을 다른 페이지에서도 사용 가능
3. **테스트 용이성**: 작은 단위로 테스트하기 쉬움
4. **유지보수성**: 특정 기능 수정 시 해당 훅만 수정하면 됨
5. **가독성**: 각 훅의 목적이 명확해짐 