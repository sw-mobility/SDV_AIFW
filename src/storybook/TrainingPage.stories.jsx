import React, { useState } from 'react';
import TrainingPage from '../pages/training_page/TrainingPage.jsx';

export default {
  title: 'Pages/TrainingPage',
  component: TrainingPage,
  parameters: {
    layout: 'fullscreen',
  },
};

export const Default = {
  render: () => <TrainingPage />, 
};

// 로딩 상태
export const Loading = {
  render: () => {
    // TrainingPage 내부의 fetchLabeledDatasets를 mock
    const [show, setShow] = useState(true);
    // 2초 후 로딩 해제
    React.useEffect(() => { const t = setTimeout(() => setShow(false), 2000); return () => clearTimeout(t); }, []);
    if (!show) return <TrainingPage />;
    // 로딩만 강제로 보여주기 위해 datasets, datasetLoading만 조작한 TrainingPage 래퍼 필요
    return (
      <div style={{ padding: 40 }}>
        <div style={{ fontSize: 18, marginBottom: 12 }}>로딩 상태 (2초 후 Default로 전환)</div>
        <TrainingPageLoadingStory />
      </div>
    );
  }
};

function TrainingPageLoadingStory() {
  // TrainingPage 내부 구조를 그대로 복제하지 않고, 주요 부분만 단순화해서 보여줌
  // 추후 fetchLabeledDatasets를 mock하는 방식으로 수정 예정
  return (
    <div style={{ padding: 40 }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: '#222', marginBottom: 10 }}>Training</div>
      <div style={{ margin: 40 }}>
        <div style={{ width: 300 }}>
          <div style={{ marginBottom: 8 }}>Dataset Loading...</div>
          <div style={{ width: 60 }}><div className="storybook-loading"><span>Loading...</span></div></div>
        </div>
      </div>
    </div>
  );
}

// 에러 상태 (데이터셋 에러)
export const DatasetError = {
  render: () => (
    <div style={{ padding: 40 }}>
      <div style={{ fontSize: 18, marginBottom: 12 }}>데이터셋 에러 상태</div>
      <TrainingPageDatasetErrorStory />
    </div>
  )
};

function TrainingPageDatasetErrorStory() {
  return (
    <div style={{ padding: 40 }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: '#222', marginBottom: 10 }}>Training</div>
      <div style={{ margin: 40 }}>
        <div style={{ width: 300 }}>
          <div style={{ marginBottom: 8, color: '#e74c3c' }}>데이터셋을 불러오는 중 에러가 발생했습니다.</div>
        </div>
      </div>
    </div>
  );
}

// IDE 모드
export const IDEMode = {
  render: () => {
    // TrainingPage의 mode를 강제로 ide로 설정하는 래퍼
    return <TrainingPageIDEModeStory />;
  }
};

function TrainingPageIDEModeStory() {
  const [mode, setMode] = useState('ide');
  // TrainingPage를 복제하지 않고, 실제로는 mode prop을 지원해야 더 깔끔함
  // 여기선 간단히 IDE 모드 UI만 보여줌
  return (
    <div style={{ padding: 40 }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: '#222', marginBottom: 10 }}>Training (IDE Mode)</div>
      <div style={{ margin: 40 }}>
        <div style={{ width: 600, border: '1px solid #eee', borderRadius: 8, padding: 16, background: '#fafbfc' }}>
          <div style={{ marginBottom: 8, fontWeight: 600 }}>Code Editor</div>
          <div style={{ height: 200, background: '#222', color: '#fff', borderRadius: 4, padding: 12 }}>코드 에디터 영역 (mock)</div>
        </div>
      </div>
    </div>
  );
}

// 학습중 상태
export const TrainingInProgress = {
  render: () => <TrainingPageTrainingStory />
};

function TrainingPageTrainingStory() {
  // 실제 TrainingPage의 isTraining, progress, logs 등을 mock
  // 여기선 간단히 진행바와 로그만 노출
  return (
    <div style={{ padding: 40 }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: '#222', marginBottom: 10 }}>Training (In Progress)</div>
      <div style={{ margin: 40 }}>
        <div style={{ marginBottom: 16 }}>
          <div style={{ width: 300, background: '#eee', borderRadius: 8, overflow: 'hidden' }}>
            <div style={{ width: '60%', height: 16, background: '#3498db' }} />
          </div>
          <div style={{ marginTop: 8, color: '#555' }}>Progress: 60%</div>
        </div>
        <div style={{ background: '#f4f4f4', borderRadius: 8, padding: 12, minHeight: 60 }}>
          <div>Training started...</div>
          <div>Progress: 10%</div>
          <div>Progress: 20%</div>
          <div>Progress: 30%</div>
          <div>Progress: 40%</div>
          <div>Progress: 50%</div>
          <div>Progress: 60%</div>
        </div>
      </div>
    </div>
  );
}

// 파라미터 에러
export const ParamError = {
  render: () => (
    <div style={{ padding: 40 }}>
      <div style={{ fontSize: 18, marginBottom: 12 }}>파라미터 에러 상태</div>
      <TrainingPageParamErrorStory />
    </div>
  )
};

function TrainingPageParamErrorStory() {
  return (
    <div style={{ padding: 40 }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: '#222', marginBottom: 10 }}>Training</div>
      <div style={{ margin: 40 }}>
        <div style={{ width: 400 }}>
          <div style={{ marginBottom: 8 }}>Epochs</div>
          <input type="number" value={0} style={{ border: '1.5px solid #e74c3c', background: '#fff6f6', width: 120, height: 32, fontSize: 16, borderRadius: 6, padding: '0 10px' }} readOnly />
          <div style={{ color: '#e74c3c', fontSize: 13, marginTop: 2, marginBottom: 4, fontWeight: 500 }}>Epochs는 1~1000 사이여야 합니다.</div>
        </div>
      </div>
    </div>
  );
} 