import { useState, useEffect, useCallback, useMemo } from 'react';
import { fetchLabeledDatasets } from '../api/datasets.js';
import { uid } from '../api/uid.js';
import { TRAINING_TYPES } from '../domain/training/trainingTypes.js';
import { getParameterGroupsByAlgorithm } from '../domain/training/parameterGroups.js';
import { validateTrainingExecution, validateParameter } from '../domain/training/trainingValidation.js';

// Mock data for snapshots
const mockSnapshots = [
  { id: 'default', name: 'Default Snapshot', description: 'System default snapshot' },
  { id: 'snap1', name: 'MyTrainerSnapshot1', description: 'Default trainer snapshot' },
  { id: 'snap2', name: 'MyTrainerSnapshot2', description: 'Experimental snapshot' },
];

// Mock snapshot files
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

export const useTrainingState = () => {
  // Core training state
  const [trainingType, setTrainingType] = useState(TRAINING_TYPES.STANDARD);
  const [algorithm, setAlgorithm] = useState('YOLO');
  const [algoParams, setAlgoParams] = useState({});
  const [paramErrors, setParamErrors] = useState({});

  // Dataset state
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [datasetError, setDatasetError] = useState(null);

  // Snapshot state
  const [snapshots, setSnapshots] = useState(mockSnapshots);
  const [selectedSnapshot, setSelectedSnapshot] = useState(null);
  const [editorFileStructure, setEditorFileStructure] = useState(snapshotFiles.default.fileStructure);
  const [editorFiles, setEditorFiles] = useState(snapshotFiles.default.files);

  // Training execution state
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('Idle');
  const [logs, setLogs] = useState([]);

  // UI state
  const [openParamGroup, setOpenParamGroup] = useState(0);
  const [showCodeEditor, setShowCodeEditor] = useState(false);
  const [selectedParamKeys, setSelectedParamKeys] = useState([]);

  // Computed values
  const paramGroups = useMemo(() => 
    getParameterGroupsByAlgorithm(algorithm), 
    [algorithm]
  );

  const trainingConfig = useMemo(() => ({
    trainingType,
    selectedDataset,
    selectedSnapshot,
    algorithm,
    algoParams
  }), [trainingType, selectedDataset, selectedSnapshot, algorithm, algoParams]);

  // Event handlers
  const handleAlgorithmChange = useCallback((newAlgorithm) => {
    setAlgorithm(newAlgorithm);
    setAlgoParams({});
    setOpenParamGroup(0);
    setSelectedParamKeys([]);
  }, []);

  const handleAlgoParamChange = useCallback((key, value, param) => {
    setAlgoParams(prev => ({ ...prev, [key]: value }));
    const { error } = validateParameter(param, value);
    setParamErrors(prev => ({ ...prev, [key]: error }));
  }, []);

  const handleToggleParamKey = useCallback((key) => {
    setSelectedParamKeys(prev =>
      prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]
    );
  }, []);

  const handleRemoveParamKey = useCallback((key) => {
    setSelectedParamKeys(prev => prev.filter(k => k !== key));
  }, []);

  const handleRunTraining = useCallback(async () => {
    const validation = validateTrainingExecution(trainingConfig);
    
    if (!validation.isValid) {
      const errorMessages = validation.errors.map(error => error.message);
      alert(errorMessages.join('\n'));
      return;
    }

    setIsTraining(true);
    setStatus('Training');
    setLogs([]);
    setProgress(0);

    try {
      // Mock training execution
      const result = await executeTraining(trainingConfig);
      
      if (result.success) {
        setLogs(prev => [...prev, result.message]);
      }
    } catch (error) {
      setLogs(prev => [...prev, `Error: ${error.message}`]);
      setIsTraining(false);
    }
  }, [trainingConfig]);

  // Effects
  useEffect(() => {
    // Fetch datasets on mount
    const fetchDatasets = async () => {
      setDatasetLoading(true);
      try {
        const res = await fetchLabeledDatasets({ uid });
        const mapped = (res.data || []).map(ds => ({
          id: ds.did || ds._id,
          name: ds.name,
          type: ds.type,
          size: ds.total,
          labelCount: ds.total,
          description: ds.description,
          task_type: ds.task_type,
          label_format: ds.label_format,
          origin_raw: ds.origin_raw,
          created_at: ds.created_at,
        }));
        setDatasets(mapped);
        setDatasetError(null);
      } catch (error) {
        setDatasetError(error.message);
      } finally {
        setDatasetLoading(false);
      }
    };

    fetchDatasets();
  }, []);

  useEffect(() => {
    // Update editor files when snapshot changes
    const snapId = selectedSnapshot ? selectedSnapshot.id : 'default';
    setEditorFileStructure(snapshotFiles[snapId]?.fileStructure || snapshotFiles.default.fileStructure);
    setEditorFiles(snapshotFiles[snapId]?.files || snapshotFiles.default.files);
  }, [selectedSnapshot]);

  useEffect(() => {
    // Training simulation
    let interval;
    if (isTraining) {
      setStatus('Training');
      setLogs(prev => [...prev, 'Training started...']);
      setProgress(0);
      
      interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            setStatus('Completed');
            setLogs(prev => [...prev, 'Training completed!']);
            setIsTraining(false);
            return 100;
          }
          setLogs(prev => [...prev, `Progress: ${prev + 10}%`]);
          return prev + 10;
        });
      }, 700);
    }
    
    return () => clearInterval(interval);
  }, [isTraining]);

  useEffect(() => {
    // Auto-complete training when progress reaches 100%
    if (progress === 100 && status !== 'success') {
      setStatus('success');
    }
  }, [progress, status]);

  return {
    // Core state
    trainingType,
    setTrainingType,
    algorithm,
    setAlgorithm,
    algoParams,
    setAlgoParams,
    paramErrors,
    setParamErrors,

    // Dataset state
    datasets,
    selectedDataset,
    setSelectedDataset,
    datasetLoading,
    datasetError,

    // Snapshot state
    snapshots,
    selectedSnapshot,
    setSelectedSnapshot,
    editorFileStructure,
    editorFiles,

    // Training execution state
    isTraining,
    setIsTraining,
    progress,
    setProgress,
    status,
    setStatus,
    logs,
    setLogs,

    // UI state
    openParamGroup,
    setOpenParamGroup,
    showCodeEditor,
    setShowCodeEditor,
    selectedParamKeys,
    setSelectedParamKeys,

    // Computed values
    paramGroups,
    trainingConfig,

    // Event handlers
    handleAlgorithmChange,
    handleAlgoParamChange,
    handleToggleParamKey,
    handleRemoveParamKey,
    handleRunTraining,
  };
};

// Mock training execution
const executeTraining = async (trainingConfig) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        success: true,
        message: 'Training started successfully',
        trainingId: `train_${Date.now()}`
      });
    }, 1000);
  });
}; 