# Frontend Architecture Guide

**Index Page**를 예로 들어 재사용 가능한 컴포넌트와 훅들을 어떻게 조립하는지 단계별로 설명

## 전체 구조 이해

### 폴더 구조
```
src/
├── components/          # 재사용 가능한 UI 컴포넌트들
│   ├── ui/             # 기본 UI 컴포넌트 (버튼, 카드, 테이블 등)
│   ├── layout/         # 레이아웃 컴포넌트 (헤더, 사이드바, 푸터)
│   └── features/       # 기능별 컴포넌트 (데이터셋, 트레이닝 등)
├── hooks/              # 비즈니스 로직을 담당하는 훅들
├── pages/              # 실제 페이지 컴포넌트들
└── api/                # 서버와 통신하는 함수들
```

## Index Page 구성 예시

### 0단계:

모든 페이지는 appRoutes 에서 불러옵니다.

```javascript
const AppRoutes = () => (<Router>
   <Routes>
      <Route path="/" element={<IndexPage/>} 
      <Route path="/projects/:projectName" element={<MainLayout/>}>
         <Route index element={<ProjectHomePage/>}/>
         <Route path="training" element={<TrainingPage/>}/>
         <Route path="labeling" element={<LabelingPage/>}/>
         <Route path="optimization" element={<OptimizationPage/>}/>
         <Route path="validation" element={<ValidationPage/>}/>
         <Route path="deployment" element={<DeploymentPage/>}/>
         <Route path="service-process" element={<ServiceProcessPage/>}/>
      </Route>


      <Route path="*" element={<Navigate to="/" replace/>}/>
   </Routes>
</Router>);

export default AppRoutes;
```

### 1단계: 페이지 컴포넌트 생성

먼저 `src/pages/index_page/IndexPage.jsx` 파일을 만듭니다.:

```javascript
import React from 'react';
import MainLayout from '../../components/layout/MainLayout.jsx';
import IndexPage from './IndexPage.jsx';
import styles from './IndexPage.module.css';

export default function IndexPage() {
  return (
    <MainLayout>
      <div className={styles.container}>
        <h1>Welcome to AI Platform</h1>
        {/* 여기에 다른 컴포넌트들을 추가할 예정 */}
      </div>
    </MainLayout>
  );
}
```

### 2단계: 필요한 hook 찾기

Index Page에서 필요한 기능들:
- 프로젝트 목록 표시
- 데이터셋 목록 표시
- 탭 전환 기능

이를 위해 다음 훅들을 사용할 수 있습니다:
- `useProjects` - 프로젝트 관련 로직
- `useDatasets` - 데이터셋 관련 로직
- `useIndexTabs` - 탭 전환 로직

### 3단계: hook import 및 사용

```javascript
import React from 'react';
import MainLayout from '../../components/layout/MainLayout.jsx';
import { useProjects } from '../../hooks/index/useProjects.js';
import { useDatasets } from '../../hooks/index/useDatasets.js';
import { useIndexTabs } from '../../hooks/index/useIndexTabs.js';
import styles from './IndexPage.module.css';

export default function IndexPage() {
  // 훅들을 사용해서 데이터와 함수들을 가져옴
  const { 
    projects, 
    loading: projectsLoading, 
    error: projectsError,
    handleCreateProject 
  } = useProjects();

  const { 
    datasets, 
    loading: datasetsLoading, 
    error: datasetsError 
  } = useDatasets();

  const { 
    activeTab, 
    setActiveTab 
  } = useIndexTabs();

  return (
    <MainLayout>
      <div className={styles.container}>
        <h1>Welcome to AI Platform</h1>
        {/* 여기에 컴포넌트들을 추가할 예정 */}
      </div>
    </MainLayout>
  );
}
```

### 4단계: UI 컴포넌트 import 및 사용

이제 UI component들을 import해서 사용합니다:

```javascript
import React from 'react';
import MainLayout from '../../components/layout/MainLayout.jsx';
import TabNavigation from '../../components/ui/TabNavigation.jsx';
import Card from '../../components/ui/Card.jsx';
import Button from '../../components/ui/Button.jsx';
import Loading from '../../components/ui/Loading.jsx';
import { useProjects } from '../../hooks/index/useProjects.js';
import { useDatasets } from '../../hooks/index/useDatasets.js';
import { useIndexTabs } from '../../hooks/index/useIndexTabs.js';
import styles from './IndexPage.module.css';

export default function IndexPage() {
  // 훅들 사용
  const { projects, loading: projectsLoading, handleCreateProject } = useProjects();
  const { datasets, loading: datasetsLoading } = useDatasets();
  const { activeTab, setActiveTab } = useIndexTabs();

  // 탭 옵션 정의
  const tabOptions = [
    { value: 'projects', label: 'Projects' },
    { value: 'datasets', label: 'Datasets' }
  ];

  return (
    <MainLayout>
      <div className={styles.container}>
        <h1>Welcome to AI Platform</h1>
        
        {/* 탭 네비게이션 */}
        <TabNavigation 
          options={tabOptions}
          value={activeTab}
          onChange={setActiveTab}
        />

        {/* 로딩 상태 처리 */}
        {projectsLoading || datasetsLoading ? (
          <Loading />
        ) : (
          <>
            {/* 프로젝트 탭 */}
            {activeTab === 'projects' && (
              <div className={styles.projectsGrid}>
                {projects.map(project => (
                  <Card key={project.id}>
                    <h3>{project.name}</h3>
                    <p>{project.description}</p>
                  </Card>
                ))}
                <Button onClick={handleCreateProject}>
                  Create New Project
                </Button>
              </div>
            )}

            {/* 데이터셋 탭 */}
            {activeTab === 'datasets' && (
              <div className={styles.datasetsGrid}>
                {datasets.map(dataset => (
                  <Card key={dataset.id}>
                    <h3>{dataset.name}</h3>
                    <p>{dataset.type}</p>
                  </Card>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </MainLayout>
  );
}
```

### 5단계: 기능별 컴포넌트로 분리

코드가 길어지면 기능별로 컴포넌트를 분리합니다:

```javascript
// src/pages/index_page/index_tab/ProjectsTab.jsx
import React from 'react';
import Card from '../../../components/ui/Card.jsx';
import Button from '../../../components/ui/Button.jsx';
import { useProjects } from '../../../hooks/index/useProjects.js';

export default function ProjectsTab() {
  const { projects, handleCreateProject } = useProjects();

  return (
    <div className={styles.projectsGrid}>
      {projects.map(project => (
        <Card key={project.id}>
          <h3>{project.name}</h3>
          <p>{project.description}</p>
        </Card>
      ))}
      <Button onClick={handleCreateProject}>
        Create New Project
      </Button>
    </div>
  );
}
```

```javascript
// src/pages/index_page/index_tab/DatasetsTab.jsx
import React from 'react';
import Card from '../../../components/ui/Card.jsx';
import { useDatasets } from '../../../hooks/index/useDatasets.js';

export default function DatasetsTab() {
  const { datasets } = useDatasets();

  return (
    <div className={styles.datasetsGrid}>
      {datasets.map(dataset => (
        <Card key={dataset.id}>
          <h3>{dataset.name}</h3>
          <p>{dataset.type}</p>
        </Card>
      ))}
    </div>
  );
}
```

그리고 메인 페이지에서 이들을 사용:

```javascript
import React from 'react';
import MainLayout from '../../components/layout/MainLayout.jsx';
import TabNavigation from '../../components/ui/TabNavigation.jsx';
import Loading from '../../components/ui/Loading.jsx';
import ProjectsTab from './index_tab/ProjectsTab.jsx';
import DatasetsTab from './index_tab/DatasetsTab.jsx';
import { useIndexTabs } from '../../hooks/index/useIndexTabs.js';
import styles from './IndexPage.module.css';

export default function IndexPage() {
  const { activeTab, setActiveTab } = useIndexTabs();

  const tabOptions = [
    { value: 'projects', label: 'Projects' },
    { value: 'datasets', label: 'Datasets' }
  ];

  return (
    <MainLayout>
      <div className={styles.container}>
        <h1>Welcome to AI Platform</h1>
        
        <TabNavigation 
          options={tabOptions}
          value={activeTab}
          onChange={setActiveTab}
        />

        {activeTab === 'projects' && <ProjectsTab />}
        {activeTab === 'datasets' && <DatasetsTab />}
      </div>
    </MainLayout>
  );
}
```

## 코드 읽는 순서

### 1. 페이지 컴포넌트부터 시작
```
src/pages/index_page/IndexPage.jsx
```
- 어떤 기능이 필요한지 파악
- 어떤 훅들을 사용할지 결정

### 2. 필요한 훅들 확인
```
src/hooks/index/useProjects.js
src/hooks/index/useDatasets.js
src/hooks/index/useIndexTabs.js
```
- 각 훅이 어떤 데이터와 함수를 제공하는지 확인
- 비즈니스 로직이 어떻게 구현되어 있는지 파악

### 3. 필요한 UI 컴포넌트들 확인
```
src/components/ui/TabNavigation.jsx
src/components/ui/Card.jsx
src/components/ui/Button.jsx
```
- 각 컴포넌트가 어떤 props를 받는지 확인
- 어떻게 렌더링되는지 파악

### 4. 기능별 컴포넌트들 확인
```
src/pages/index_page/index_tab/ProjectsTab.jsx
src/pages/index_page/index_tab/DatasetsTab.jsx
```
- 각 탭이 어떻게 구현되어 있는지 확인

## 조립 패턴

### 패턴 1: 페이지 → 훅 → UI 컴포넌트
```javascript
// 1. 페이지에서 훅 사용
const { data, loading, handleAction } = useSomeHook();

// 2. UI 컴포넌트에 데이터 전달
<SomeComponent 
  data={data}
  loading={loading}
  onAction={handleAction}
/>
```

### 패턴 2: 페이지 → 기능별 컴포넌트 → 훅
```javascript
// 1. 페이지에서 기능별 컴포넌트 사용
<ProjectsTab />

// 2. 기능별 컴포넌트에서 훅 사용
const { projects } = useProjects();
```

### 패턴 3: 레이아웃 컴포넌트로 감싸기
```javascript
// 모든 페이지는 MainLayout으로 감싸기
<MainLayout>
  <PageContent />
</MainLayout>
```

### 관심사 분리
- **훅**: 비즈니스 로직 (데이터 가져오기, 상태 관리)
- **컴포넌트**: UI 렌더링 (화면에 표시)
- **페이지**: 조립 (훅과 컴포넌트를 연결)

## 실제 개발 순서

1. **기능 요구사항 파악**
   - 어떤 데이터가 필요한가?
   - 어떤 동작이 필요한가?

2. **훅 선택/생성**
   - 기존 훅이 있는지 확인
   - 없다면 새로 생성

3. **UI 컴포넌트 선택/생성**
   - 기존 컴포넌트가 있는지 확인
   - 없다면 새로 생성

4. **페이지에서 조립**
   - 훅에서 데이터 가져오기
   - 컴포넌트에 데이터 전달

5. **스타일링**
   - CSS 모듈로 스타일 적용
