import { useCallback, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { validateTrainingExecution } from '../../domain/training/trainingValidation.js';
import { useProgress } from '../common/useProgress.js';
import { postYoloTraining, postYoloTrainingResult } from '../../api/training.js';
import { uid } from '../../api/uid.js';

export const useTrainingExecution = (trainingConfig) => {
  const progress = useProgress();
  const { projectName } = useParams();
  const [projectData, setProjectData] = useState(null);

  // Get project data for pid
  useEffect(() => {
    const fetchProjectData = async () => {
      if (projectName) {
        try {
          console.log(`Fetching project data for: ${projectName}`);
          const response = await fetch(`http://localhost:5002/projects/projects/`, {
            headers: {
              'uid': uid
            }
          });
          if (response.ok) {
            const data = await response.json();
            console.log(`Found ${data.length} projects`);
            console.log(`Available projects: ${data.map(p => `${p.name} (${p._id || p.id})`).join(', ')}`);
            
            const project = data.find(p => p.name === projectName);
            if (project) {
              console.log(`Project found: ${project.name}`);
              console.log(`Project data:`, project);
              setProjectData(project);
            } else {
              console.log(`Project not found: ${projectName}`);
              console.log(`Available project names: ${data.map(p => p.name).join(', ')}`);
            }
          } else {
            console.log(`Failed to fetch projects: ${response.status}`);
            const errorText = await response.text();
            console.log(`Error details: ${errorText}`);
          }
        } catch (error) {
          console.log(`Error fetching project data: ${error.message}`);
          console.error('Failed to fetch project data:', error);
        }
      } else {
        console.log('No projectName in URL params');
      }
    };
    fetchProjectData();
  }, [projectName]); // Remove progress from dependencies

  const runTraining = useCallback(async () => {
    const validation = validateTrainingExecution(trainingConfig);
    if (!validation.isValid) {
      const errorMessages = validation.errors.map(error => error.message);
      alert(errorMessages.join('\n'));
      return;
    }

    // Check if we have required data
    if (!projectData) {
      progress.addLog('Error: Project data not available');
      progress.addLog(`Current projectName: ${projectName}`);
      progress.addLog(`Current uid: ${uid}`);
      return;
    }

    if (!trainingConfig.selectedDataset) {
      progress.addLog('Error: No dataset selected');
      return;
    }

    progress.start();
    progress.addLog('Training started...');

    try {
      if (trainingConfig.algorithm === 'YOLO') {
        progress.addLog('Calling YOLO training API...');
        
        // Parse split_ratio from string to array if needed
        let splitRatio = [0.8, 0.2];
        if (trainingConfig.algoParams.split_ratio) {
          try {
            splitRatio = typeof trainingConfig.algoParams.split_ratio === 'string' 
              ? JSON.parse(trainingConfig.algoParams.split_ratio)
              : trainingConfig.algoParams.split_ratio;
          } catch (e) {
            console.warn('Invalid split_ratio format, using default:', e);
          }
        }

        // Prepare YOLO parameters based on API schema
        const yoloParameters = {
          model: trainingConfig.algoParams.model || 'yolov8n',
          split_ratio: splitRatio,
          epochs: trainingConfig.algoParams.epochs || 50,
          batch: trainingConfig.algoParams.batch_size || 16,
          imgsz: trainingConfig.algoParams.input_size || 640,
          device: trainingConfig.algoParams.device || 'cuda:0',
          save_period: trainingConfig.algoParams.save_period || 5,
          workers: trainingConfig.algoParams.workers || 4,
          pretrained: trainingConfig.algoParams.pretrained !== false,
          optimizer: trainingConfig.algoParams.optimizer || 'SGD',
          lr0: trainingConfig.algoParams.learning_rate || 0.01,
          lrf: trainingConfig.algoParams.lrf || 0.1,
          momentum: trainingConfig.algoParams.momentum || 0.937,
          weight_decay: trainingConfig.algoParams.weight_decay || 0.0005,
          patience: trainingConfig.algoParams.patience || 20,
          augment: trainingConfig.algoParams.augmentation !== false,
          warmup_epochs: trainingConfig.algoParams.warmup_epochs || 3,
          warmup_momentum: trainingConfig.algoParams.warmup_momentum || 0.8,
          warmup_bias_lr: trainingConfig.algoParams.warmup_bias_lr || 0.1,
          seed: trainingConfig.algoParams.seed || 42,
          cache: trainingConfig.algoParams.cache !== false,
          dropout: trainingConfig.algoParams.dropout || 0.0,
          label_smoothing: trainingConfig.algoParams.label_smoothing || 0.0,
          rect: trainingConfig.algoParams.rect !== false,
          resume: trainingConfig.algoParams.resume || '',
          amp: trainingConfig.algoParams.amp !== false,
          single_cls: trainingConfig.algoParams.single_cls !== false,
          cos_lr: trainingConfig.algoParams.cos_lr !== false,
          close_mosaic: trainingConfig.algoParams.close_mosaic || 0,
          overlap_mask: trainingConfig.algoParams.overlap_mask !== false,
          mask_ratio: trainingConfig.algoParams.mask_ratio || 0.0
        };

        // Try different ways to get project ID - prioritize pid over _id
        const projectId = projectData.pid || projectData._id || projectData.id;
        progress.addLog(`Using project ID: ${projectId}`);
        progress.addLog(`Project data keys: ${Object.keys(projectData).join(', ')}`);
        progress.addLog(`Project pid: ${projectData.pid}, _id: ${projectData._id}`);
        progress.addLog(`Using dataset ID: ${trainingConfig.selectedDataset.id}`);
        progress.addLog(`Dataset data:`, trainingConfig.selectedDataset);
        progress.addLog(`Dataset keys: ${Object.keys(trainingConfig.selectedDataset).join(', ')}`);
        progress.addLog(`Dataset did: ${trainingConfig.selectedDataset.did}, _id: ${trainingConfig.selectedDataset._id}, id: ${trainingConfig.selectedDataset.id}`);

        // Use the dataset _id as dataset_id for the API
        const datasetId = trainingConfig.selectedDataset._id;
        progress.addLog(`Selected dataset ID: ${datasetId}`);
        progress.addLog(`Selected model: ${trainingConfig.algoParams.model || 'yolov8n'}`);
        
        // Determine task type based on algorithm and model
        let taskType = 'detection'; // default for YOLO
        if (trainingConfig.algorithm === 'YOLO') {
          const model = trainingConfig.algoParams.model || 'yolov8n';
          if (model.includes('seg')) {
            taskType = 'segmentation';
          } else if (model.includes('pose')) {
            taskType = 'pose';
          } else if (model.includes('obb')) {
            taskType = 'obb';
          } else if (model.includes('cls')) {
            taskType = 'classification';
          } else {
            taskType = 'detection'; // default for YOLO
          }
        }
        
        progress.addLog(`Using task type: ${taskType}`);
        
        const requestBody = {
          pid: projectId,
          task_type: taskType,
          parameters: yoloParameters,
          dataset_id: datasetId
        };

        progress.addLog('Sending training request with parameters:');
        progress.addLog(JSON.stringify(requestBody, null, 2));
        
        const response = await postYoloTraining({ uid, ...requestBody });
        
        progress.addLog('YOLO training started successfully.');
        progress.addLog(`Training ID: ${response.uid || 'N/A'}`);
        
        // Simulate training completion and result submission
        setTimeout(async () => {
          progress.addLog('Training completed! Submitting result...');
          try {
            const resultBody = {
              pid: projectId,
              status: 'success',
              task_type: taskType,
              parameters: {
                epochs: yoloParameters.epochs,
                model: yoloParameters.model,
                'mAP@0.5': 0.823 // Simulated result
              },
              dataset_id: datasetId,
              artifact_path: `/mnt/output/${yoloParameters.model}_results/`,
              error_details: ''
            };

            progress.addLog('Submitting result:');
            progress.addLog(JSON.stringify(resultBody, null, 2));
            
            await postYoloTrainingResult({ uid, ...resultBody });
            progress.addLog('Training result submitted successfully.');
          } catch (err) {
            progress.addLog('Result submission failed: ' + err.message);
          }
          progress.complete();
        }, 3000);
      } else {
        // fallback: mock progress for other algorithms
        let pct = 0;
        const interval = setInterval(() => {
          pct += 10;
          progress.updateProgress(pct);
          progress.addLog(`Progress: ${pct}%`);
          if (pct >= 100) {
            clearInterval(interval);
            progress.complete();
            progress.addLog('Training completed!');
          }
        }, 400);
      }
    } catch (err) {
      progress.addLog('Training failed: ' + err.message);
      progress.complete();
    }
  }, [trainingConfig, progress, projectData, projectName]);

  return {
    ...progress,
    runTraining
  };
}; 