import React, { useState, useEffect } from 'react';
import styles from './OptimizationPage.module.css';
import TargetBoardSelector from '../../components/optimization/TargetBoardSelector.jsx';
import ModelSelector from '../../components/optimization/ModelSelector.jsx';
import OptionEditor from '../../components/optimization/OptionEditor.jsx';
import ProgressBar from '../../components/common/ProgressBar.jsx';
import DatasetTablePanel from '../../components/labeling/DatasetTablePanel.jsx';
import Table from '../../components/common/Table.jsx';
import { fetchLabeledDatasets, fetchRawDatasets } from '../../api/datasets.js';
import { uid } from '../../api/uid.js';
import useOptimizationState from '../../hooks/useOptimizationState.js';

const parameterDefs = [
  { key: 'learning_rate', label: 'Learning Rate', type: 'number', default: 0.001 },
  { key: 'batch_size', label: 'Batch Size', type: 'number', default: 16 },
  { key: 'optimizer', label: 'Optimizer', type: 'select', options: ['QAT', 'Adam', 'SGD'] },
  { key: 'use_int8', label: 'Use INT8', type: 'checkbox' }
];

const OptimizationPage = () => {
  const {
    targetBoard, setTargetBoard,
    model, setModel,
    options, setOptions,
    isRunning, progress, logs, status,
    runOptimization,
    trials
  } = useOptimizationState();

  const [labeledDatasets, setLabeledDatasets] = useState([]);
  const [loadingLabeled, setLoadingLabeled] = useState(false);
  const [rawDatasets, setRawDatasets] = useState([]);
  const [loadingRaw, setLoadingRaw] = useState(false);
  const [calibrationDataset, setCalibrationDataset] = useState(null);

  useEffect(() => {
    setLoadingLabeled(true);
    fetchLabeledDatasets({ uid }).then(res => {
      setLabeledDatasets(res.data || []);
    }).finally(() => setLoadingLabeled(false));
  }, []);

  useEffect(() => {
    setLoadingRaw(true);
    fetchRawDatasets({ uid }).then(res => {
      setRawDatasets(res.data || []);
    }).finally(() => setLoadingRaw(false));
  }, []);

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
          <div className={styles.selectorBlock}>
            <TargetBoardSelector value={targetBoard} onChange={setTargetBoard} disabled={isRunning} />
          </div>
          <div className={styles.selectorBlock}>
            <DatasetTablePanel
              datasets={rawDatasets}
              selectedId={calibrationDataset?.id}
              onSelect={setCalibrationDataset}
              loading={loadingRaw}
              title="Calibration Dataset"
            />
          </div>
          <div className={styles.selectorBlock}>
            <ModelSelector value={model} onChange={setModel} disabled={isRunning} />
          </div>
        </div>
        <div className={styles.rightPanel}>
          <div className={styles.optionCardContainer}>
            <OptionEditor
              options={options}
              onChange={setOptions}
              onRun={runOptimization}
              isRunning={isRunning}
              parameterDefs={parameterDefs}
            />
            <div className={styles.progressSection}>
              <div className={styles.statusBadge}>
                {status === 'idle' && <span className={styles.statusIdle}>Ready</span>}
                {status === 'running' && <span className={styles.statusRunning}>Running</span>}
                {status === 'success' && <span className={styles.statusSuccess}>Completed</span>}
                {status === 'error' && <span className={styles.statusError}>Error</span>}
              </div>
              <div style={{ margin: '16px 0' }}>
                <ProgressBar value={progress} status={status} />
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
            <div className={styles.trialSection}>
              <Table
                columns={['Trial', 'Metric', 'Params']}
                data={(trials || []).map((t, i) => [i+1, t.metric, t.paramSummary])}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
export default OptimizationPage;
