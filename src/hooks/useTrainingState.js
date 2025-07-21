import { useState, useEffect } from 'react';
import { fetchLabeledDatasets } from '../api/datasets.js';

// 스냅샷 관련 상수
export const defaultSnapshot = { id: 'default', name: 'Default Snapshot', description: 'System default snapshot' };
export const mockSnapshots = [
  defaultSnapshot,
  { id: 'snap1', name: 'MyTrainerSnapshot1', description: 'Default trainer snapshot' },
  { id: 'snap2', name: 'MyTrainerSnapshot2', description: 'Experimental snapshot' },
];

// 스냅샷별 파일 구조/내용 (mock)
export const snapshotFiles = {
  default: {
    fileStructure: [
      {
        name: 'src',
        type: 'folder',
        children: [
          { name: 'train.py', type: 'file' },
          { name: 'data_loader.py', type: 'file' },
          { name: 'model_config.py', type: 'file' },
          { name: 'train_parameter.json', type: 'file' },
        ]
      }
    ],
    files: {
      'train.py': { code: '# Default train.py', language: 'python' },
      'data_loader.py': { code: '# Default data_loader.py', language: 'python' },
      'model_config.py': { code: '# Default model_config.py', language: 'python' },
      'train_parameter.json': { code: '{\n  "epochs": 20\n}', language: 'json' },
    }
  },
  snap1: {
    fileStructure: [
      {
        name: 'src',
        type: 'folder',
        children: [
          { name: 'train.py', type: 'file' },
          { name: 'data_loader.py', type: 'file' },
          { name: 'model_config.py', type: 'file' },
          { name: 'train_parameter.json', type: 'file' },
        ]
      }
    ],
    files: {
      'train.py': { code: '# snap1 train.py', language: 'python' },
      'data_loader.py': { code: '# snap1 data_loader.py', language: 'python' },
      'model_config.py': { code: '# snap1 model_config.py', language: 'python' },
      'train_parameter.json': { code: '{\n  "epochs": 30\n}', language: 'json' },
    }
  },
  snap2: {
    fileStructure: [
      {
        name: 'src',
        type: 'folder',
        children: [
          { name: 'train.py', type: 'file' },
          { name: 'data_loader.py', type: 'file' },
          { name: 'model_config.py', type: 'file' },
          { name: 'train_parameter.json', type: 'file' },
        ]
      }
    ],
    files: {
      'train.py': { code: '# snap2 train.py', language: 'python' },
      'data_loader.py': { code: '# snap2 data_loader.py', language: 'python' },
      'model_config.py': { code: '# snap2 model_config.py', language: 'python' },
      'train_parameter.json': { code: '{\n  "epochs": 50\n}', language: 'json' },
    }
  },
};

export const useTrainingState = () => {
  // Tabs & Mode
  const [trainingType, setTrainingType] = useState('standard');
  const [mode, setMode] = useState('no-code');

  // Dataset
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [datasetError, setDatasetError] = useState(null);

  // Snapshot
  const [snapshots, setSnapshots] = useState(mockSnapshots);
  const [selectedSnapshot, setSelectedSnapshot] = useState(null);
  const [snapshotModalOpen, setSnapshotModalOpen] = useState(false);

  // Algorithm
  const [algorithm, setAlgorithm] = useState('YOLO');
  const [algoParams, setAlgoParams] = useState({});
  const [paramErrors, setParamErrors] = useState({});

  // Training
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('Idle');
  const [logs, setLogs] = useState([]);

  // UI State
  const [openParamGroup, setOpenParamGroup] = useState(0);
  const [showCodeEditor, setShowCodeEditor] = useState(false);
  // 여러 파라미터를 선택할 수 있도록 배열로 관리
  const [selectedParamKeys, setSelectedParamKeys] = useState([]);
  const [editorFileStructure, setEditorFileStructure] = useState(snapshotFiles['default'].fileStructure);
  const [editorFiles, setEditorFiles] = useState(snapshotFiles['default'].files);

  // Fetch labeled datasets on mount
  useEffect(() => {
    setDatasetLoading(true);
    fetchLabeledDatasets()
      .then(res => {
        // 변환: DatasetSelector가 기대하는 필드로 매핑
        const mapped = (res.data || []).map(ds => ({
          id: ds.did || ds._id,
          name: ds.name,
          type: ds.type,
          size: ds.total,
          labelCount: ds.total, // 실제 라벨 개수 필드가 있으면 교체
          description: ds.description,
          task_type: ds.task_type,
          label_format: ds.label_format,
          origin_raw: ds.origin_raw,
          created_at: ds.created_at,
        }));
        setDatasets(mapped);
        setDatasetError(null);
      })
      .catch(e => setDatasetError(e.message))
      .finally(() => setDatasetLoading(false));
  }, []);

  // 스냅샷 선택 시 파일 구조/내용 변경
  useEffect(() => {
    const snapId = selectedSnapshot ? selectedSnapshot.id : 'default';
    setEditorFileStructure(snapshotFiles[snapId]?.fileStructure || snapshotFiles['default'].fileStructure);
    setEditorFiles(snapshotFiles[snapId]?.files || snapshotFiles['default'].files);
  }, [selectedSnapshot]);

  // Training simulation (mock)
  useEffect(() => {
    let interval;
    if (isTraining) {
      setStatus('Training');
      setLogs(l => [...l, 'Training started...']);
      setProgress(0);
      interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            setStatus('Completed');
            setLogs(l => [...l, 'Training completed!']);
            setIsTraining(false);
            return 100;
          }
          setLogs(l => [...l, `Progress: ${prev + 10}%`]);
          return prev + 10;
        });
      }, 700);
    }
    return () => clearInterval(interval);
  }, [isTraining]);

  return {
    // Training Type & Mode
    trainingType, setTrainingType,
    mode, setMode,
    
    // Dataset
    datasets, setDatasets,
    selectedDataset, setSelectedDataset,
    datasetLoading, setDatasetLoading,
    datasetError, setDatasetError,
    
    // Snapshot
    snapshots, setSnapshots,
    selectedSnapshot, setSelectedSnapshot,
    snapshotModalOpen, setSnapshotModalOpen,
    
    // Algorithm
    algorithm, setAlgorithm,
    algoParams, setAlgoParams,
    paramErrors, setParamErrors,
    
    // Training
    isTraining, setIsTraining,
    progress, setProgress,
    status, setStatus,
    logs, setLogs,
    
    // UI State
    openParamGroup, setOpenParamGroup,
    showCodeEditor, setShowCodeEditor,
    selectedParamKeys, setSelectedParamKeys,
    editorFileStructure, setEditorFileStructure,
    editorFiles, setEditorFiles,
  };
}; 