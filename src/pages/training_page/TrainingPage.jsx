import React, { useEffect, useState } from 'react';
import Button from '../../components/common/Button.jsx';
import ProgressBar from '../../components/common/ProgressBar.jsx';
import Loading from '../../components/common/Loading.jsx';
import CodeEditor from '../../components/common/CodeEditor.jsx';
import { fetchLabeledDatasets } from '../../api/datasets.js';
import pageStyles from '../index_page/IndexPage.module.css';
import styles from './TrainingPage.module.css';
import { PlayCircle, ChevronDown, ChevronUp, Plus, Minus, Edit3, HelpCircle } from 'lucide-react';
import TextField from '@mui/material/TextField';
import InputAdornment from '@mui/material/InputAdornment';
import IconButton from '@mui/material/IconButton';
import Slider from '@mui/material/Slider';
import Tooltip from '@mui/material/Tooltip';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';

const defaultSnapshot = { id: 'default', name: 'Default Snapshot', description: 'System default snapshot' };
const mockSnapshots = [
  defaultSnapshot,
  { id: 'snap1', name: 'MyTrainerSnapshot1', description: 'Default trainer snapshot' },
  { id: 'snap2', name: 'MyTrainerSnapshot2', description: 'Experimental snapshot' },
];

// 스냅샷별 파일 구조/내용 (mock)
const snapshotFiles = {
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

// 파라미터 요약 컴포넌트
function ParamSummary({ paramGroups, algoParams }) {
  if (!paramGroups) return null;
  return (
    <div className={styles.paramSummaryGroupWrap}>
      {paramGroups.map((group, gidx) => (
        <div key={group.group} className={styles.paramSummaryGroupBox} data-group={group.group}>
          <div className={styles.paramGroupTitle}>{group.group}</div>
          {group.params.map((param, idx) => {
            const value = algoParams[param.key] ?? param.default;
            const isChanged = value !== param.default;
            return (
              <div key={param.key} className={styles.paramSummaryItem}>
                <span className={styles.paramSummaryLabel}>{param.label}</span>
                <span className={isChanged ? styles.paramSummaryValue : styles.paramSummaryValueDefault}>{String(value)}</span>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

// 2. ParamChipList component (replaces ParamSummary)
function ParamChipList({ paramGroups, algoParams, selectedKey, onSelect }) {
  if (!paramGroups) return null;
  return (
    <div className={styles.paramChipListWrap}>
      {paramGroups.flatMap((group) =>
        group.params.map((param) => {
          const value = algoParams[param.key] ?? param.default;
          const isChanged = value !== param.default;
          const isSelected = selectedKey === param.key;
          return (
            <button
              key={param.key}
              className={
                styles.paramChip +
                (isChanged ? ' ' + styles.paramChipChanged : ' ' + styles.paramChipDefault) +
                (isSelected ? ' ' + styles.paramChipSelected : '')
              }
              onClick={() => onSelect(param.key)}
              type="button"
              tabIndex={0}
            >
              <span className={styles.paramChipLabel}>{param.label}</span>
              <span className={styles.paramChipValue}>{String(value)}</span>
            </button>
          );
        })
      )}
    </div>
  );
}

export default function TrainingPage() {
  // Tabs & Mode
  const [trainingType, setTrainingType] = useState('standard'); // 'standard' | 'continual'
  const [mode, setMode] = useState('no-code'); // 'no-code' | 'ide'

  // Dataset
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [datasetError, setDatasetError] = useState(null);

  // Snapshot
  const [snapshots, setSnapshots] = useState(mockSnapshots);
  const [selectedSnapshot, setSelectedSnapshot] = useState(null);
  const [snapshotModalOpen, setSnapshotModalOpen] = useState(false);

  // Hyperparameters
  const [params, setParams] = useState({ epochs: 20, lr: 0.01, batchSize: 32 });
  const [paramErrors, setParamErrors] = useState({ epochs: '', lr: '', batchSize: '' });

  // Training
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('Idle');
  const [logs, setLogs] = useState([]);

  // Algorithm
  const algorithmOptions = [
    { value: 'yolov8', label: 'YOLOv8' },
    { value: 'algorithm1', label: 'Algorithm 1' },
    { value: 'algorithm2', label: 'Algorithm 2' },
  ];
  const yolov8ParamGroups = [
    {
      group: 'Data Split',
      params: [
        { key: 'train_ratio', label: 'Train Ratio', type: 'number', min: 0, max: 1, step: 0.01, default: 0.7 },
        { key: 'val_ratio', label: 'Val Ratio', type: 'number', min: 0, max: 1, step: 0.01, default: 0.2 },
        { key: 'test_ratio', label: 'Test Ratio', type: 'number', min: 0, max: 1, step: 0.01, default: 0.1 },
      ],
    },
    {
      group: 'Training',
      params: [
        { key: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 1000, step: 1, default: 20 },
        { key: 'batch_size', label: 'Batch Size', type: 'number', min: 1, max: 1024, step: 1, default: 32 },
        { key: 'imgsz', label: 'Image Size', type: 'number', min: 32, max: 4096, step: 32, default: 640 },
        { key: 'device', label: 'Device', type: 'text', default: 'cpu' },
        { key: 'patience', label: 'Patience', type: 'number', min: 0, max: 100, step: 1, default: 50 },
        { key: 'augment', label: 'Augment', type: 'checkbox', default: true },
        { key: 'pretrained', label: 'Pretrained', type: 'checkbox', default: true },
      ],
    },
    {
      group: 'Optimization',
      params: [
        { key: 'optimizer', label: 'Optimizer', type: 'select', options: ['SGD', 'Adam', 'AdamW'], default: 'SGD' },
        { key: 'lr0', label: 'Initial LR', type: 'number', min: 0.00001, max: 1, step: 0.00001, default: 0.01 },
        { key: 'lrf', label: 'LR Final', type: 'number', min: 0.00001, max: 1, step: 0.00001, default: 0.01 },
        { key: 'momentum', label: 'Momentum', type: 'number', min: 0, max: 1, step: 0.01, default: 0.937 },
        { key: 'weight_decay', label: 'Weight Decay', type: 'number', min: 0, max: 1, step: 0.0001, default: 0.0005 },
        { key: 'warmup_epochs', label: 'Warmup Epochs', type: 'number', min: 0, max: 100, step: 1, default: 3 },
        { key: 'warmup_momentum', label: 'Warmup Momentum', type: 'number', min: 0, max: 1, step: 0.01, default: 0.8 },
        { key: 'warmup_bias_lr', label: 'Warmup Bias LR', type: 'number', min: 0, max: 1, step: 0.0001, default: 0.1 },
      ],
    },
    {
      group: 'Model/Experiment Management',
      params: [
        { key: 'model', label: 'Model', type: 'text', default: 'yolov8n.pt' },
        { key: 'project', label: 'Project', type: 'text', default: 'runs/train' },
        { key: 'name', label: 'Name', type: 'text', default: '' },
      ],
    },
  ];
  const algorithm1Params = [
    { key: 'paramA', label: 'Param A', type: 'number', min: 0, max: 100, step: 1, default: 10 },
    { key: 'paramB', label: 'Param B', type: 'text', default: '' },
  ];
  const algorithm2Params = [
    { key: 'alpha', label: 'Alpha', type: 'number', min: 0, max: 1, step: 0.01, default: 0.5 },
  ];

  // Algorithm state
  const [algorithm, setAlgorithm] = useState('yolov8');
  const [algoParams, setAlgoParams] = useState({});

  // Accordion open state
  const [openParamGroup, setOpenParamGroup] = useState(0);

  // Code Editor toggle state
  const [showCodeEditor, setShowCodeEditor] = useState(false);
  // 코드 에디터용 상태: 현재 파일 구조/내용 (스냅샷 선택에 따라 변경)
  const [editorFileStructure, setEditorFileStructure] = useState(snapshotFiles['default'].fileStructure);
  const [editorFiles, setEditorFiles] = useState(snapshotFiles['default'].files);

  // 스냅샷 선택 시 파일 구조/내용 변경
  useEffect(() => {
    const snapId = selectedSnapshot ? selectedSnapshot.id : 'default';
    setEditorFileStructure(snapshotFiles[snapId]?.fileStructure || snapshotFiles['default'].fileStructure);
    setEditorFiles(snapshotFiles[snapId]?.files || snapshotFiles['default'].files);
  }, [selectedSnapshot]);

  // Drawer close handler
  const handleCloseDrawer = () => setShowCodeEditor(false);

  // Fetch labeled datasets on mount
  useEffect(() => {
    setDatasetLoading(true);
    fetchLabeledDatasets()
      .then(res => {
        setDatasets(res.data);
        setDatasetError(null);
      })
      .catch(e => setDatasetError(e.message))
      .finally(() => setDatasetLoading(false));
  }, []);

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

  // Handlers
  const handleRunTraining = () => {
    if (mode === 'no-code' && (!selectedDataset || !selectedSnapshot)) {
      alert('Please select both a dataset and a snapshot.');
      return;
    }
    setIsTraining(true);
    setStatus('Training');
    setLogs([]);
    setProgress(0);
  };

  // --- validateParam 범용화 및 에러 관리 개선 ---
  const validateParam = (param, value) => {
    let error = '';
    if (param.type === 'number') {
      if (typeof value !== 'number' || isNaN(value)) {
        error = '숫자를 입력하세요.';
      } else if (param.min !== undefined && value < param.min) {
        error = `${param.label}은(는) 최소 ${param.min} 이상이어야 합니다.`;
      } else if (param.max !== undefined && value > param.max) {
        error = `${param.label}은(는) 최대 ${param.max} 이하여야 합니다.`;
      }
    } else if (param.type === 'text') {
      if (param.required && (!value || value === '')) {
        error = `${param.label}을(를) 입력하세요.`;
      }
    }
    setParamErrors(prev => ({ ...prev, [param.key]: error }));
    return error === '';
  };

  // --- handleAlgoParamChange에서 항상 validateParam 호출 ---
  const handleAlgoParamChange = (key, value, param) => {
    let newValue = value;
    if (param && param.type === 'number') {
      const decimals = getDecimalPlaces(param.step);
      newValue = Number(Number(value).toFixed(decimals));
    }
    setAlgoParams(p => ({ ...p, [key]: newValue }));
    validateParam(param, newValue);
  };

  // step에서 소수점 자릿수 계산
  const getDecimalPlaces = (step) => {
    if (!step || step >= 1) return 0;
    return step.toString().split('.')[1]?.length || 0;
  };

  // 파라미터 입력 필드 스타일 통일 (TextField/select/input)
  const inputSizeStyle = { width: 150, height: 38, fontSize: 15, fontWeight: 500, borderRadius: 6, background: '#fff', minWidth: 0, maxWidth: '100%' };

  // --- 파라미터 입력 필드 렌더링 개선 (에러 UX 통일) ---
  // (yolov8, algorithm1, algorithm2 모두 동일하게 적용)
  // Add state for selected parameter key
  const [selectedParamKey, setSelectedParamKey] = useState(null);

  // Helper to get current param definition
  const getCurrentParam = () => {
    const groups = algorithm === 'yolov8' ? yolov8ParamGroups : algorithm === 'algorithm1' ? [{ group: 'Algorithm 1', params: algorithm1Params }] : [{ group: 'Algorithm 2', params: algorithm2Params }];
    for (const group of groups) {
      for (const param of group.params) {
        if (param.key === selectedParamKey) return { ...param, group: group.group };
      }
    }
    return null;
  };
  const currentParam = getCurrentParam();

  return (
    <div className={styles.container}>
      <div style={{ fontSize: 28, fontWeight: 700, color: "#222", marginBottom: "10px"}}>Training</div>
      {/* Standard/Continual Training Tab Navigation */}
      <div className={styles.tabNavigation}>
        <button
          className={`${pageStyles.tabButton} ${trainingType === 'standard' ? pageStyles.activeTab : ''}`}
          onClick={() => setTrainingType('standard')}
        >
          Standard Training
        </button>
        <button
          className={`${pageStyles.tabButton} ${trainingType === 'continual' ? pageStyles.activeTab : ''}`}
          onClick={() => setTrainingType('continual')}
        >
          Continual Training
        </button>
      </div>
      <hr className={styles.tabDivider} />
      {/* Algorithm Selector (always shown) */}
      <div className={styles.sectionCard}>
        <div className={styles.selectorGroup}>
          <div className={styles.selectorBox}>
            <label className={styles.paramLabel} style={{marginBottom: 4}}>Algorithm</label>
            <select
              className={styles.select}
              value={algorithm}
              onChange={e => {
                setAlgorithm(e.target.value);
                setAlgoParams({});
                setOpenParamGroup(0);
              }}
            >
              {algorithmOptions.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
        </div>
        {/* Edit Code/Expert Mode button */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '16px' }}>
          <Button
              variant="secondary"
              onClick={() => setShowCodeEditor(true)}
              style={{ minWidth: 140 }}
          >
            Edit Code (Expert Mode)
          </Button>
        </div>
      </div>
      {/* Standard/Continual UI */}
      {trainingType === 'standard' ? (
          <>
          {/* Dataset & Snapshot (Standard) */}
          <div className={styles.sectionCard}>
            <div className={styles.selectorGroup}>
              <div className={styles.selectorBox}>
                <label className={styles.paramLabel} style={{marginBottom: 4}}>Dataset</label>
                {datasetLoading && <Loading />}
                {datasetError && <span className={styles.inputErrorMsg}>{datasetError}</span>}
                {!datasetLoading && !datasetError && (
                  <select
                    className={styles.select}
                    value={selectedDataset ? selectedDataset.id : ''}
                    onChange={e => {
                      const ds = datasets.find(d => d.id === Number(e.target.value));
                      setSelectedDataset(ds);
                    }}
                  >
                    <option value="">Select dataset</option>
                    {datasets.map(ds => (
                      <option key={ds.id} value={ds.id}>
                        {ds.name} ({ds.type}, {ds.size})
                      </option>
                    ))}
                  </select>
                )}
                {selectedDataset && (
                  <div className={styles.datasetInfo}>
                    <div><b>Name:</b> {selectedDataset.name}</div>
                    <div><b>Type:</b> {selectedDataset.type}</div>
                    <div><b>Size:</b> {selectedDataset.size}</div>
                    <div><b>Label Count:</b> {selectedDataset.labelCount}</div>
                  </div>
                )}
              </div>
              <div className={styles.selectorBox}>
                <label className={styles.paramLabel} style={{marginBottom: 4}}>Snapshot</label>
                <div className={styles.snapshotRow}>
                  <select
                    className={styles.select}
                    value={selectedSnapshot ? selectedSnapshot.id : 'default'}
                    onChange={e => {
                      const snap = snapshots.find(s => s.id === e.target.value) || defaultSnapshot;
                      setSelectedSnapshot(snap.id === 'default' ? null : snap);
                    }}
                  >
                    {snapshots.map(snap => (
                      <option key={snap.id} value={snap.id}>{snap.name}</option>
                    ))}
                  </select>
                </div>
                {selectedSnapshot && (
                  <div className={styles.snapshotInfo}>
                    <div><b>Name:</b> {selectedSnapshot.name}</div>
                    <div><b>Description:</b> {selectedSnapshot.description}</div>
                  </div>
                )}
              </div>
            </div>
          </div>
          {/* Parameters & Summary 2단 레이아웃 (Summary 왼쪽, 입력 오른쪽) */}
          <div className={styles.paramSectionWrap}>
            {/* Left: ParamChipList */}
            <div className={styles.paramSummaryBox + ' ' + styles.sectionCard}>
              <div className={styles.paramGroupTitle} style={{ fontSize: 17, marginBottom: 12 }}>Parameters</div>
              <ParamChipList
                paramGroups={algorithm === 'yolov8' ? yolov8ParamGroups : algorithm === 'algorithm1' ? [{ group: 'Algorithm 1', params: algorithm1Params }] : [{ group: 'Algorithm 2', params: algorithm2Params }]}
                algoParams={algoParams}
                selectedKey={selectedParamKey}
                onSelect={setSelectedParamKey}
              />
            </div>
            {/* Right: Parameter Form */}
            <div className={styles.paramCardWrap}>
              {currentParam ? (
                <div className={styles.paramCard + ' ' + styles.sectionCard + ' ' + styles.paramCardActive}>
                  <div className={styles.paramRowHeader}>
                    <span className={styles.paramLabel}>{currentParam.label}</span>
                    <Tooltip title={`기본값: ${currentParam.default}${currentParam.min !== undefined ? `, 범위: ${currentParam.min}~${currentParam.max}` : ''}`.trim()} placement="right">
                      <HelpCircle size={18} color="#888" style={{ marginLeft: 6, verticalAlign: 'middle' }} />
                    </Tooltip>
                  </div>
                  {/* Input field by type */}
                  {currentParam.type === 'number' ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                      <Slider
                        min={currentParam.min}
                        max={currentParam.max}
                        step={currentParam.step}
                        value={algoParams[currentParam.key] ?? currentParam.default}
                        onChange={(_, v) => handleAlgoParamChange(currentParam.key, v, currentParam)}
                        sx={{ width: 180, color: '#4f8cff' }}
                      />
                      <input
                        type="number"
                        value={algoParams[currentParam.key] ?? currentParam.default}
                        min={currentParam.min}
                        max={currentParam.max}
                        step={currentParam.step}
                        onChange={e => handleAlgoParamChange(currentParam.key, Number(e.target.value), currentParam)}
                        className={styles.paramInput}
                        style={{ width: 80, marginLeft: 8 }}
                      />
                    </div>
                  ) : currentParam.type === 'select' ? (
                    <select
                      className={styles.paramInput}
                      value={algoParams[currentParam.key] ?? currentParam.default}
                      onChange={e => handleAlgoParamChange(currentParam.key, e.target.value, currentParam)}
                      style={{ width: 180 }}
                    >
                      {currentParam.options.map(opt => (
                        <option key={opt} value={opt}>{opt}</option>
                      ))}
                    </select>
                  ) : currentParam.type === 'checkbox' ? (
                    <div className={styles.switchContainer}>
                      <Switch
                        checked={algoParams[currentParam.key] ?? currentParam.default}
                        onChange={e => handleAlgoParamChange(currentParam.key, e.target.checked, currentParam)}
                        color="primary"
                        size="medium"
                      />
                    </div>
                  ) : (
                    <input
                      type="text"
                      className={styles.paramInput}
                      value={algoParams[currentParam.key] ?? currentParam.default}
                      onChange={e => handleAlgoParamChange(currentParam.key, e.target.value, currentParam)}
                      style={{ width: 180 }}
                    />
                  )}
                  {currentParam.desc && <div className={styles.paramDesc}>{currentParam.desc}</div>}
                  {/* Reset button removed for cleaner UI */}
                </div>
              ) : (
                <div className={styles.paramCard + ' ' + styles.sectionCard + ' ' + styles.paramCardEmpty}>
                  <span style={{ color: '#aaa', fontSize: 15 }}>왼쪽에서 파라미터를 선택하세요.</span>
                </div>
              )}
            </div>
          </div>
          </>
        ) : (
          <>
          {/* Continual Learning Info */}
          <div className={styles.sectionCard} style={{ background: '#f8f9fb', border: '1.5px solid #4f8cff', color: '#1a3a5d', marginBottom: 18 }}>
            <b>Continual Learning</b> allows you to update a model incrementally with new data, starting from a previous snapshot. Select a base snapshot and a new dataset to continue training.
          </div>
          {/* Base Snapshot (required) and New Dataset */}
          <div className={styles.sectionCard}>
            <div className={styles.selectorGroup}>
              <div className={styles.selectorBox}>
                <label className={styles.paramLabel} style={{marginBottom: 4}}>Base Snapshot <span style={{color:'#e74c3c'}}>*</span></label>
                <div className={styles.snapshotRow}>
                  <select
                    className={styles.select}
                    value={selectedSnapshot ? selectedSnapshot.id : 'default'}
                    onChange={e => {
                      const snap = snapshots.find(s => s.id === e.target.value) || defaultSnapshot;
                      setSelectedSnapshot(snap.id === 'default' ? null : snap);
                    }}
                  >
                    {snapshots.map(snap => (
                      <option key={snap.id} value={snap.id}>{snap.name}</option>
                    ))}
                  </select>
                </div>
                {selectedSnapshot && (
                  <div className={styles.snapshotInfo}>
                    <div><b>Name:</b> {selectedSnapshot.name}</div>
                    <div><b>Description:</b> {selectedSnapshot.description}</div>
                  </div>
                )}
              </div>
              <div className={styles.selectorBox}>
                <label className={styles.paramLabel} style={{marginBottom: 4}}>New Dataset <span style={{color:'#e74c3c'}}>*</span></label>
                {datasetLoading && <Loading />}
                {datasetError && <span className={styles.inputErrorMsg}>{datasetError}</span>}
                {!datasetLoading && !datasetError && (
                  <select
                    className={styles.select}
                    value={selectedDataset ? selectedDataset.id : ''}
                    onChange={e => {
                      const ds = datasets.find(d => d.id === Number(e.target.value));
                      setSelectedDataset(ds);
                    }}
                  >
                    <option value="">Select new dataset</option>
                    {datasets.map(ds => (
                      <option key={ds.id} value={ds.id}>
                        {ds.name} ({ds.type}, {ds.size})
                      </option>
                    ))}
                  </select>
                )}
                {selectedDataset && (
                  <div className={styles.datasetInfo}>
                    <div><b>Name:</b> {selectedDataset.name}</div>
                    <div><b>Type:</b> {selectedDataset.type}</div>
                    <div><b>Size:</b> {selectedDataset.size}</div>
                    <div><b>Label Count:</b> {selectedDataset.labelCount}</div>
                  </div>
                )}
              </div>
            </div>
          </div>
          {/* Parameters - Only show 'Training' group for Continual */}
          <div className={styles.paramCardWrap}>
            {algorithm === 'yolov8' ? (
              yolov8ParamGroups.filter(g => g.group === 'Training').map((group, idx) => (
                <div key={group.group} className={styles.accordionCard}>
                  <div
                    className={styles.accordionHeader + ' ' + (openParamGroup === idx ? styles.accordionOpen : '')}
                    onClick={() => setOpenParamGroup(openParamGroup === idx ? -1 : idx)}
                    tabIndex={0}
                    role="button"
                    aria-expanded={openParamGroup === idx}
                  >
                    <span>{group.group}</span>
                    <span className={styles.accordionArrow}>{openParamGroup === idx ? <ChevronUp size={18} color="#4f8cff" /> : <ChevronDown size={18} color="#4f8cff" />}</span>
                  </div>
                  {openParamGroup === idx && (
                    <div className={styles.accordionContent}>
                      {group.params.map(param => {
                        const error = paramErrors[param.key];
                        const isNumber = param.type === 'number';
                        return (
                          <div className={styles.paramRow} key={param.key}>
                            <label className={styles.paramLabel}>{param.label}</label>
                            {param.type === 'select' ? (
                              <select
                                className={styles.paramInput + (error ? ' ' + styles.inputError : '')}
                                value={algoParams[param.key] ?? param.default}
                                onChange={e => handleAlgoParamChange(param.key, e.target.value, param)}
                                disabled={isTraining}
                                style={inputSizeStyle}
                              >
                                {param.options.map(opt => (
                                  <option key={opt} value={opt}>{opt}</option>
                                ))}
                              </select>
                            ) : param.type === 'checkbox' ? (
                              <div className={styles.switchContainer}>
                                <Switch
                                  checked={algoParams[param.key] ?? param.default}
                                  onChange={e => handleAlgoParamChange(param.key, e.target.checked, param)}
                                  disabled={isTraining}
                                  color="primary"
                                  size="medium"
                                />
                                <span className={styles.switchLabel}>활성화</span>
                              </div>
                            ) : isNumber ? (
                              <TextField
                                type="number"
                                size="small"
                                value={algoParams[param.key] ?? param.default}
                                onChange={e => handleAlgoParamChange(param.key, Number(e.target.value), param)}
                                inputProps={{
                                  min: param.min,
                                  max: param.max,
                                  step: param.step,
                                  style: { textAlign: 'right', fontSize: 15, fontWeight: 500, padding: '6px 10px' }
                                }}
                                InputProps={{
                                  endAdornment: (
                                    <InputAdornment position="end" sx={{ ml: 0 }}>
                                      <IconButton size="small" onClick={() => { const v = Number(Number((algoParams[param.key] ?? param.default) + (param.step || 1)).toFixed(getDecimalPlaces(param.step))); handleAlgoParamChange(param.key, v, param); }} disabled={isTraining || (algoParams[param.key] ?? param.default) >= param.max}>
                                        <Plus size={16} />
                                      </IconButton>
                                      <IconButton size="small" onClick={() => { const v = Number(Number((algoParams[param.key] ?? param.default) - (param.step || 1)).toFixed(getDecimalPlaces(param.step))); handleAlgoParamChange(param.key, v, param); }} disabled={isTraining || (algoParams[param.key] ?? param.default) <= param.min}>
                                        <Minus size={16} />
                                      </IconButton>
                                    </InputAdornment>
                                  ),
                                  sx: {
                                    borderRadius: 2,
                                    background: '#fff',
                                    minWidth: 0,
                                    maxWidth: '100%',
                                  }
                                }}
                                sx={{ ...inputSizeStyle, ...(error ? { border: '1.5px solid #e74c3c', background: '#fff6f6' } : {}) }}
                                error={!!error}
                                disabled={isTraining}
                                helperText={error || ' '}
                                FormHelperTextProps={{ style: { marginTop: 4, minHeight: 18, fontSize: 13, fontWeight: 500, color: '#e74c3c', background: 'none' } }}
                              />
                            ) : (
                              <input
                                type={param.type}
                                className={styles.paramInput + (error ? ' ' + styles.inputError : '')}
                                value={algoParams[param.key] ?? param.default}
                                min={param.min}
                                max={param.max}
                                step={param.step}
                                placeholder={param.label}
                                disabled={isTraining}
                                onChange={e => handleAlgoParamChange(param.key, param.type === 'number' ? Number(e.target.value) : e.target.value, param)}
                                style={inputSizeStyle}
                              />
                            )}
                            {param.desc && <span className={styles.paramDesc}>{param.desc}</span>}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              ))
            ) : algorithm === 'algorithm1' ? (
              <div className={styles.accordionCard}>
                <div
                  className={styles.accordionHeader + ' ' + (openParamGroup === 0 ? styles.accordionOpen : '')}
                  onClick={() => setOpenParamGroup(openParamGroup === 0 ? -1 : 0)}
                  tabIndex={0}
                  role="button"
                  aria-expanded={openParamGroup === 0}
                >
                  <span>Algorithm 1</span>
                  <span className={styles.accordionArrow}>{openParamGroup === 0 ? <ChevronUp size={18} color="#4f8cff" /> : <ChevronDown size={18} color="#4f8cff" />}</span>
                </div>
                {openParamGroup === 0 && (
                  <div className={styles.accordionContent}>
                    {algorithm1Params.map(param => {
                      const error = paramErrors[param.key];
                      const isNumber = param.type === 'number';
                      return (
                        <div className={styles.paramRow} key={param.key}>
                          <label className={styles.paramLabel}>{param.label}</label>
                          {isNumber ? (
                            <TextField
                              type="number"
                              size="small"
                              value={algoParams[param.key] ?? param.default}
                              onChange={e => handleAlgoParamChange(param.key, Number(e.target.value), param)}
                              inputProps={{
                                min: param.min,
                                max: param.max,
                                step: param.step,
                                style: { textAlign: 'right', fontSize: 15, fontWeight: 500, padding: '6px 10px' }
                              }}
                              InputProps={{
                                endAdornment: (
                                  <InputAdornment position="end" sx={{ ml: 0 }}>
                                    <IconButton size="small" onClick={() => { const v = Number(Number((algoParams[param.key] ?? param.default) + (param.step || 1)).toFixed(getDecimalPlaces(param.step))); handleAlgoParamChange(param.key, v, param); }} disabled={isTraining || (algoParams[param.key] ?? param.default) >= param.max}>
                                      <Plus size={16} />
                                    </IconButton>
                                    <IconButton size="small" onClick={() => { const v = Number(Number((algoParams[param.key] ?? param.default) - (param.step || 1)).toFixed(getDecimalPlaces(param.step))); handleAlgoParamChange(param.key, v, param); }} disabled={isTraining || (algoParams[param.key] ?? param.default) <= param.min}>
                                      <Minus size={16} />
                                    </IconButton>
                                  </InputAdornment>
                                ),
                                sx: {
                                  borderRadius: 2,
                                  background: '#fff',
                                  minWidth: 0,
                                  maxWidth: '100%',
                                }
                              }}
                              sx={{ ...inputSizeStyle, ...(error ? { border: '1.5px solid #e74c3c', background: '#fff6f6' } : {}) }}
                              error={!!error}
                              disabled={isTraining}
                              helperText={error || ' '}
                              FormHelperTextProps={{ style: { marginTop: 4, minHeight: 18, fontSize: 13, fontWeight: 500, color: '#e74c3c', background: 'none' } }}
                            />
                          ) : (
                            <input
                              type={param.type}
                              className={styles.paramInput + (error ? ' ' + styles.inputError : '')}
                              value={algoParams[param.key] ?? param.default}
                              min={param.min}
                              max={param.max}
                              step={param.step}
                              placeholder={param.label}
                              disabled={isTraining}
                              onChange={e => handleAlgoParamChange(param.key, param.type === 'number' ? Number(e.target.value) : e.target.value, param)}
                              style={inputSizeStyle}
                            />
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            ) : algorithm === 'algorithm2' ? (
              <div className={styles.accordionCard}>
                <div
                  className={styles.accordionHeader + ' ' + (openParamGroup === 0 ? styles.accordionOpen : '')}
                  onClick={() => setOpenParamGroup(openParamGroup === 0 ? -1 : 0)}
                  tabIndex={0}
                  role="button"
                  aria-expanded={openParamGroup === 0}
                >
                  <span>Algorithm 2</span>
                  <span className={styles.accordionArrow}>{openParamGroup === 0 ? <ChevronUp size={18} color="#4f8cff" /> : <ChevronDown size={18} color="#4f8cff" />}</span>
                </div>
                {openParamGroup === 0 && (
                  <div className={styles.accordionContent}>
                    {algorithm2Params.map(param => {
                      const error = paramErrors[param.key];
                      const isNumber = param.type === 'number';
                      return (
                        <div className={styles.paramRow} key={param.key}>
                          <label className={styles.paramLabel}>{param.label}</label>
                          {isNumber ? (
                            <TextField
                              type="number"
                              size="small"
                              value={algoParams[param.key] ?? param.default}
                              onChange={e => handleAlgoParamChange(param.key, Number(e.target.value), param)}
                              inputProps={{
                                min: param.min,
                                max: param.max,
                                step: param.step,
                                style: { textAlign: 'right', fontSize: 15, fontWeight: 500, padding: '6px 10px' }
                              }}
                              InputProps={{
                                endAdornment: (
                                  <InputAdornment position="end" sx={{ ml: 0 }}>
                                    <IconButton size="small" onClick={() => { const v = Number(Number((algoParams[param.key] ?? param.default) + (param.step || 1)).toFixed(getDecimalPlaces(param.step))); handleAlgoParamChange(param.key, v, param); }} disabled={isTraining || (algoParams[param.key] ?? param.default) >= param.max}>
                                      <Plus size={16} />
                                    </IconButton>
                                    <IconButton size="small" onClick={() => { const v = Number(Number((algoParams[param.key] ?? param.default) - (param.step || 1)).toFixed(getDecimalPlaces(param.step))); handleAlgoParamChange(param.key, v, param); }} disabled={isTraining || (algoParams[param.key] ?? param.default) <= param.min}>
                                      <Minus size={16} />
                                    </IconButton>
                                  </InputAdornment>
                                ),
                                sx: {
                                  borderRadius: 2,
                                  background: '#fff',
                                  minWidth: 0,
                                  maxWidth: '100%',
                                }
                              }}
                              sx={{ ...inputSizeStyle, ...(error ? { border: '1.5px solid #e74c3c', background: '#fff6f6' } : {}) }}
                              error={!!error}
                              disabled={isTraining}
                              helperText={error || ' '}
                              FormHelperTextProps={{ style: { marginTop: 4, minHeight: 18, fontSize: 13, fontWeight: 500, color: '#e74c3c', background: 'none' } }}
                            />
                          ) : (
                            <input
                              type={param.type}
                              className={styles.paramInput + (error ? ' ' + styles.inputError : '')}
                              value={algoParams[param.key] ?? param.default}
                              min={param.min}
                              max={param.max}
                              step={param.step}
                              placeholder={param.label}
                              disabled={isTraining}
                              onChange={e => handleAlgoParamChange(param.key, param.type === 'number' ? Number(e.target.value) : e.target.value, param)}
                              style={inputSizeStyle}
                            />
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </>
      )}

      {/* Drawer for Code Editor */}
      {showCodeEditor && (
          <>
            <div className={styles.drawerOverlay} onClick={handleCloseDrawer}></div>
            <div className={styles.codeDrawer}>
              <div className={styles.drawerEditorWrap}>
                <CodeEditor
                    snapshotName={selectedSnapshot ? selectedSnapshot.name : 'Default Snapshot'}
                    fileStructure={editorFileStructure}
                    files={editorFiles}
                    // activeFile, onFileChange, onFilesChange 등은 추후 필요시 추가
                    onSaveSnapshot={name => {
                      // 실제 저장 로직은 추후 구현 (현재는 alert)
                      alert(`Saved as snapshot: ${name}`);
                    }}
                    onCloseDrawer={handleCloseDrawer}
                />
              </div>
            </div>
          </>
      )}
      {/* Run Section */}
      <div className={styles.sectionCard}>
        <div className={styles.runCard}>
          <div className={styles.runRow}>
            <Button
                variant="primary-gradient"
                size="medium"
                onClick={handleRunTraining}
                disabled={isTraining}
                icon={isTraining ? <span className={styles.spinner}></span> : <PlayCircle size={20} style={{ marginRight: 6 }} />}
                style={{ minWidth: 150, opacity: isTraining ? 0.6 : 1, cursor: isTraining ? 'not-allowed' : 'pointer' }}
            >
              {isTraining ? 'Running...' : 'Run Training'}
            </Button>
          </div>
        </div>
        <div className={styles.statusCard}>
          <div>
            <ProgressBar percentage={progress} />
          </div>
          <div className={styles.logBox}>
            {logs.length === 0 ? (
                <span className={styles.logEmpty}>No logs yet.</span>
            ) : (
                logs.map((log, i) => <div key={i}>{log}</div>)
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
