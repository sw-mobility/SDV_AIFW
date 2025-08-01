import React, { useState, useEffect } from 'react';
import styles from './OptimizationPage.module.css';
import TargetBoardSelector from '../../components/features/optimization/TargetBoardSelector.jsx';
import ModelSelector from '../../components/features/optimization/ModelSelector.jsx';
import OptionEditor from '../../components/features/optimization/OptionEditor.jsx';
import ProgressBar from '../../components/ui/atoms/ProgressBar.jsx';
import DatasetTablePanel from '../../components/features/labeling/DatasetTablePanel.jsx';
import { fetchLabeledDatasets, fetchRawDatasets } from '../../api/datasets.js';
import { uid } from '../../api/uid.js';
import useOptimizationState from '../../hooks/index.js';

const parameterDefsMap = {
  'board1:modelA': [
    { key: 'learning_rate', label: 'Learning Rate', type: 'number', default: 0.001 },
    { key: 'batch_size', label: 'Batch Size', type: 'number', default: 16 },
    { key: 'optimizer', label: 'Optimizer', type: 'select', options: ['QAT', 'Adam', 'SGD'] },
    { key: 'use_int8', label: 'Use INT8', type: 'checkbox' }
  ],
  'board2:modelB': [
    { key: 'momentum', label: 'Momentum', type: 'number', default: 0.9 },
    { key: 'weight_decay', label: 'Weight Decay', type: 'number', default: 0.0005 }
  ],
  'default': [
    { key: 'learning_rate', label: 'Learning Rate', type: 'number', default: 0.001 },
    { key: 'batch_size', label: 'Batch Size', type: 'number', default: 16 }
  ]
};

const getParameterDefs = (board, model) => {
  if (parameterDefsMap[`${board}:${model}`]) return parameterDefsMap[`${board}:${model}`];
  return parameterDefsMap['default'];
};

const OptimizationPage = () => {
  const {
    targetBoard, setTargetBoard,
    model, setModel,
    options, setOptions,
    isRunning, progress, logs, status,
    runOptimization, setTestDataset
  } = useOptimizationState();

  const [labeledDatasets, setLabeledDatasets] = useState([]);
  const [loadingLabeled, setLoadingLabeled] = useState(false);
  const [rawDatasets, setRawDatasets] = useState([]);
  const [loadingRaw, setLoadingRaw] = useState(false);
  const [calibrationDataset, setCalibrationDataset] = useState(null);

  // calibrationDataset이 바뀔 때마다 testDataset도 동기화
  useEffect(() => {
    if (calibrationDataset && calibrationDataset.id) {
      setTestDataset(calibrationDataset.id);
    } else {
      setTestDataset('');
    }
  }, [calibrationDataset, setTestDataset]);

  useEffect(() => {
    setLoadingLabeled(true);
    fetchLabeledDatasets({ uid }).then(res => {
      setLabeledDatasets(res.data || []);
    }).finally(() => setLoadingLabeled(false));
  }, []);

  useEffect(() => {
    setLoadingRaw(true);
    fetchRawDatasets({ uid }).then(res => {
      const camelDatasets = (res.data || []).map(ds => ({
        ...ds,
        id: ds.id || ds._id, // id 필드 보장
        createdAt: ds.created_at ? new Date(ds.created_at).toISOString().slice(0, 10) : undefined
      }));
      setRawDatasets(camelDatasets);
    }).finally(() => setLoadingRaw(false));
  }, []);

  const handleRunOptimization = () => {
    if (!targetBoard || !model || !calibrationDataset) {
      setOptions(prev => ({ ...prev }));
      return;
    }
    runOptimization();
  };

  return (
    <div className={styles.container}>
      <div className={styles.pageHeader}>
        <h1 className={styles.pageTitle}>Optimization</h1>
        <p className={styles.pageDescription}>
          Configure your optimization settings and run model optimization with your selected board, calibration dataset, and model.
        </p>
      </div>
      <div className={styles.sectionWrap}>
        <div className={styles.leftPanel}>
          <div className={`${styles.selectorBlock} ${styles.noBorder}`}>
            <TargetBoardSelector value={targetBoard} onChange={setTargetBoard} disabled={isRunning} />
          </div>
            <DatasetTablePanel
              datasets={rawDatasets}
              selectedId={calibrationDataset?.id}
              onSelect={setCalibrationDataset}
              loading={loadingRaw}
            />
          <div className={`${styles.selectorBlock} ${styles.noBorder}`} style={{marginTop: '24px'}}>
            <ModelSelector value={model} onChange={setModel} disabled={isRunning} />
          </div>
        </div>
        <div className={styles.rightPanel}>
          <div className={styles.card}>
            <OptionEditor
              options={options}
              onChange={setOptions}
              onRun={handleRunOptimization}
              isRunning={isRunning}
              parameterDefs={getParameterDefs(targetBoard, model)}
              targetBoard={targetBoard}
              model={model}
            />
          </div>
          <div className={styles.card}>
            <div className={styles.progressSection}>
              <div className={styles.statusBadge}>
                {status === 'idle' && <span className={styles.statusIdle}>Ready</span>}
                {status === 'running' && <span className={styles.statusRunning}>Running</span>}
                {status === 'success' && <span className={styles.statusSuccess}>Completed</span>}
                {status === 'error' && <span className={styles.statusError}>Error</span>}
              </div>
              <div style={{ margin: '16px 0' }}>
                <ProgressBar percentage={progress} status={status} />
              </div>
              {logs && logs.length > 0 && (
                <div className={styles.logSection}>
                  <h4 className={styles.logTitle}>Logs</h4>
                  <div className={styles.logArea}>
                    {logs.map((log, idx) => (
                      <div key={idx} className={styles.logLine}>{log}</div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
export default OptimizationPage;
