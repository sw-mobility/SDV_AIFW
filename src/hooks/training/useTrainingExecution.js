import { useCallback, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { validateTrainingExecution } from '../../domain/training/trainingValidation.js';
import { useProgress } from '../common/useProgress.js';
import { postYoloTraining } from '../../api/training.js';
import { uid } from '../../api/uid.js';

export const useTrainingExecution = (trainingConfig) => {
  const progress = useProgress();
  const { projectName } = useParams();
  const [projectData, setProjectData] = useState(null);
  const [trainingResponse, setTrainingResponse] = useState(null);

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
    console.log('=== Training Execution Started ===');
    console.log('Training config:', trainingConfig);
    
    const validation = validateTrainingExecution(trainingConfig);
    console.log('Validation result:', validation);
    
    if (!validation.isValid) {
      const errorMessages = validation.errors.map(error => error.message);
      console.log('Validation errors:', errorMessages);
      alert(errorMessages.join('\n'));
      return;
    }

    // Check if we have required data
    console.log('Project data:', projectData);
    console.log('Selected dataset:', trainingConfig.selectedDataset);
    
    if (!projectData) {
      progress.addLog('Error: Project data not available');
      progress.addLog(`Current projectName: ${projectName}`);
      progress.addLog(`Current uid: ${uid}`);
      console.log('Training stopped: No project data');
      return;
    }

    if (!trainingConfig.selectedDataset) {
      progress.addLog('Error: No dataset selected');
      console.log('Training stopped: No dataset selected');
      return;
    }

    progress.start();
    progress.addLog('Training started...');

    try {
      console.log('Algorithm:', trainingConfig.algorithm);
      if (trainingConfig.algorithm === 'YOLO' || trainingConfig.algorithm === 'yolo_v8') {
        progress.addLog('Calling YOLO training API...');
        console.log('Starting YOLO training...');
        
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

        // Parse COCO classes from YAML to string list for API
        let userClasses = [];
        if (trainingConfig.algoParams.coco_classes) {
          try {
            const lines = trainingConfig.algoParams.coco_classes.split('\n');
            let inNamesSection = false;
            
            for (const line of lines) {
              const trimmed = line.trim();
              if (trimmed === 'names:') {
                inNamesSection = true;
                continue;
              }
              
              if (inNamesSection && trimmed.includes(':')) {
                const [key, value] = trimmed.split(':').map(s => s.trim());
                if (/^\d+$/.test(key) && value) {
                  userClasses.push(value);  // String list 형태로 저장: ["class1", "class2"]
                }
              }
            }
            
            progress.addLog(`Parsed ${userClasses.length} classes from COCO configuration`);
            progress.addLog(`Classes: ${JSON.stringify(userClasses)}`);
          } catch (error) {
            progress.addLog(`Warning: Failed to parse COCO classes: ${error.message}`);
            progress.addLog('Using default classes configuration');
          }
        }

        // Prepare YOLO parameters based on API schema
        const yoloParameters = {
          model: trainingConfig.algoParams.model || 'yolov8n',
          split_ratio: splitRatio,
          epochs: trainingConfig.algoParams.epochs || 50,
          batch: trainingConfig.algoParams.batch_size || 16,
          imgsz: trainingConfig.algoParams.input_size || 640,
          device: trainingConfig.algoParams.device || 'cpu',
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
          dropout: trainingConfig.algoParams.dropout || 0,
          label_smoothing: trainingConfig.algoParams.label_smoothing || 0,
          rect: trainingConfig.algoParams.rect !== false,
          resume: trainingConfig.algoParams.resume || '',
          amp: trainingConfig.algoParams.amp !== false,
          single_cls: trainingConfig.algoParams.single_cls !== false,
          cos_lr: trainingConfig.algoParams.cos_lr !== false,
          close_mosaic: trainingConfig.algoParams.close_mosaic || 0,
          overlap_mask: trainingConfig.algoParams.overlap_mask !== false
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

        // Use the dataset did as dataset_id for the API
        const datasetId = trainingConfig.selectedDataset.did;
        progress.addLog(`Selected dataset ID: ${datasetId}`);
        progress.addLog(`Selected model: ${trainingConfig.algoParams.model || 'yolov8n'}`);
        
        // Check if dataset has classes
        const datasetClasses = trainingConfig.selectedDataset.classes || [];
        progress.addLog(`Dataset classes: ${JSON.stringify(datasetClasses)}`);
        progress.addLog(`Dataset classes length: ${datasetClasses.length}`);
        
        // Determine task type based on algorithm and model
        let taskType = 'detection'; // default for YOLO
        if (trainingConfig.algorithm === 'YOLO' || trainingConfig.algorithm === 'yolo_v8') {
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
        
        // Use dataset classes if no custom classes are provided
        const finalUserClasses = userClasses.length > 0 ? userClasses : 
          (datasetClasses.length > 0 ? datasetClasses : undefined);
        
        const requestBody = {
          pid: projectId,
          did: datasetId,
          user_classes: finalUserClasses,
          parameters: yoloParameters
        };

        console.log('Final request body:', requestBody);
        console.log('Dataset ID:', datasetId);
        console.log('User classes:', userClasses);
        console.log('User classes length:', userClasses.length);
        
        progress.addLog('Sending training request with parameters:');
        progress.addLog(JSON.stringify(requestBody, null, 2));
        
        console.log('About to call postYoloTraining API...');
        const response = await postYoloTraining({ uid, ...requestBody });
        console.log('API response:', response);
        
        // Save training response
        setTrainingResponse(response);
        
        progress.addLog('YOLO training started successfully.');
        progress.addLog(`Training ID: ${response.data?.tid || response.tid || 'N/A'}`);
        progress.complete();
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
    runTraining,
    trainingResponse
  };
}; 