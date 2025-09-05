import { useCallback, useState } from 'react';
import { validateTrainingExecution } from '../../domain/training/trainingValidation.js';
import { useProgress } from '../common/useProgress.js';
import { postYoloTraining} from '../../api/training.js';
import { uid } from '../../api/uid.js';

export const useTrainingExecution = (trainingConfig) => {
  const progress = useProgress();
  const [trainingResponse, setTrainingResponse] = useState(null);

  const runTraining = useCallback(async () => {
    console.log('=== Training Execution Start ===');
    console.log('trainingConfig:', trainingConfig);
    
    const validation = validateTrainingExecution(trainingConfig);
    console.log('validation result:', validation);
    
    if (!validation.isValid) {
      const errorMessages = validation.errors.map(error => error.message);
      console.error('Validation failed:', errorMessages);
      alert(errorMessages.join('\n'));
      return;
    }

    if (!trainingConfig.selectedDataset) {
      progress.addLog('Error: No dataset selected');
      return;
    }

    progress.start();
    progress.addLog('Training started...');

    try {
      // Algorithm 체크를 더 유연하게 수정
      const isYoloAlgorithm = trainingConfig.algorithm && 
        (trainingConfig.algorithm.toLowerCase().includes('yolo') || 
         trainingConfig.algorithm.toLowerCase().includes('yolo_v'));
      
      if (isYoloAlgorithm) {
        progress.addLog(`Starting ${trainingConfig.algorithm} training...`);
        
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
        
        console.log('=== Training Config Debug ===');
        console.log('trainingConfig:', trainingConfig);
        console.log('trainingConfig.algoParams keys:', Object.keys(trainingConfig.algoParams || {}));
        console.log('coco_classes value:', trainingConfig.algoParams?.coco_classes);
        console.log('coco_classes type:', typeof trainingConfig.algoParams?.coco_classes);
        console.log('coco_classes length:', trainingConfig.algoParams?.coco_classes?.length);
        
        if (trainingConfig.algoParams?.coco_classes) {
          try {
            const yamlContent = trainingConfig.algoParams.coco_classes.trim();
            console.log('Raw coco_classes input:', yamlContent);
            console.log('Input length:', yamlContent.length);
            console.log('Input type:', typeof yamlContent);
            
            // Check if input is empty or just whitespace
            if (!yamlContent || yamlContent.length === 0) {
              console.log('Input is empty or whitespace only');
            } else {
              const lines = yamlContent.split('\n');
              console.log('Total lines:', lines.length);
              let inNamesSection = false;
              
              for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                const trimmed = line.trim();
                console.log(`Line ${i}:`, `"${trimmed}"`);
                
                if (trimmed === 'names:') {
                  inNamesSection = true;
                  console.log('Found names section at line', i);
                  continue;
                }
                
                if (inNamesSection && trimmed.includes(':')) {
                  const [key, value] = trimmed.split(':').map(s => s.trim());
                  console.log('Parsing key-value:', key, '->', value);
                  if (/^\d+$/.test(key) && value) {
                    userClasses.push(value);
                    console.log('Added class:', value);
                  }
                }
              }
            }
            
            console.log('Final parsed user classes:', userClasses);
          } catch (error) {
            console.warn('Failed to parse COCO classes:', error);
            console.error('Error details:', error);
          }
        } else {
          console.log('No coco_classes provided in trainingConfig.algoParams');
          
          // Use dataset classes if available
          if (trainingConfig.selectedDataset?.classes) {
            userClasses = trainingConfig.selectedDataset.classes;
            console.log('Using dataset classes:', userClasses);
          } else {
            // Fallback to default classes
            const defaultClasses = ['person', 'bicycle', 'car'];
            console.log('Using default classes:', defaultClasses);
            userClasses = defaultClasses;
          }
        }
        
        console.log('Final userClasses after parsing:', userClasses);

        // Device 값을 API 형식으로 변환
        const convertDeviceForAPI = (deviceValue) => {
          console.log('Converting device value:', deviceValue);
          let apiDevice;
          
          if (deviceValue === '0') {
            apiDevice = '0'; // GPU 0
          } else if (deviceValue === '1') {
            apiDevice = '1'; // GPU 1
          } else {
            apiDevice = deviceValue; // cpu는 그대로
          }
          
          console.log('Converted device for API:', apiDevice);
          return apiDevice;
        };

        // Prepare YOLO parameters based on API schema
        const yoloParameters = {
          model: trainingConfig.algoParams.model || 'yolov8n',
          split_ratio: splitRatio,
          epochs: trainingConfig.algoParams.epochs || 50,
          batch: trainingConfig.algoParams.batch_size || 16,
          imgsz: trainingConfig.algoParams.input_size || 640,
          device: convertDeviceForAPI(trainingConfig.algoParams.device || '0'),
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
          resume: trainingConfig.algoParams.resume === true,
          amp: trainingConfig.algoParams.amp !== false,
          single_cls: trainingConfig.algoParams.single_cls !== false,
          cos_lr: trainingConfig.algoParams.cos_lr !== false,
          close_mosaic: trainingConfig.algoParams.close_mosaic || 0,
          overlap_mask: trainingConfig.algoParams.overlap_mask !== false
        };

        // Get project and dataset IDs from trainingConfig
        const projectId = trainingConfig.projectId || 'P0001'; // TrainingPage에서 전달받은 projectId 사용
        const datasetId = trainingConfig.selectedDataset.did;
        const datasetClasses = trainingConfig.selectedDataset.classes || [];
        
        console.log('=== Dataset Info ===');
        console.log('selectedDataset:', trainingConfig.selectedDataset);
        console.log('datasetId:', datasetId);
        console.log('projectId:', projectId);
        
        // Check if using custom model
        const isCustomModel = trainingConfig.modelType === 'custom';
        const customModelId = trainingConfig.customModel;
        
        if (isCustomModel && !customModelId) {
          throw new Error('Custom model을 선택해주세요.');
        }
        
        // Convert dataset classes to string list if needed
        const datasetClassesList = Array.isArray(datasetClasses) 
          ? datasetClasses.map(cls => typeof cls === 'string' ? cls : cls.name || cls.toString())
          : [];
        
        // Always prioritize user input over dataset classes
        let finalUserClasses = userClasses.length > 0 ? userClasses : 
          (datasetClassesList.length > 0 ? datasetClassesList : []);
        
        // If user provided custom classes, force them to be used
        if (userClasses.length > 0) {
          finalUserClasses = userClasses;
          console.log('Forcing user-defined classes:', userClasses);
        } else {
          // Use dataset classes if available, otherwise default
          if (datasetClassesList.length > 0) {
            finalUserClasses = datasetClassesList;
            console.log('Using dataset classes:', datasetClassesList);
          } else {
            finalUserClasses = ['person', 'bicycle', 'car'];
            console.log('Using default classes:', finalUserClasses);
          }
        }
        
        // Log which classes are being used
        console.log('=== Classes Debug ===');
        console.log('User input classes:', userClasses);
        console.log('Dataset classes:', datasetClassesList);
        console.log('Final classes being sent:', finalUserClasses);
        
        const requestBody = {
          pid: projectId,
          did: datasetId, // API에서 did 필드 요구
          user_classes: finalUserClasses
        };

        // codebase가 선택된 경우 cid 필드 추가
        if (trainingConfig.selectedCodebase?.cid) {
          requestBody.cid = trainingConfig.selectedCodebase.cid;
          console.log('Adding cid to request:', trainingConfig.selectedCodebase.cid);
        }
        
        console.log('=== Request Body Before Model Type Check ===');
        console.log('requestBody:', requestBody);
        
        // Model type에 따라 다른 방식으로 request body 구성
        if (isCustomModel) {
          // Custom model일 때: origin_tid 사용, parameters에서 model 제거
          requestBody.origin_tid = customModelId;
          
          // parameters에서 model을 제거한 새로운 parameters 객체 생성
          const { model, ...parametersWithoutModel } = yoloParameters;
          requestBody.parameters = parametersWithoutModel;
          
          console.log('Using custom model with origin_tid:', customModelId);
          console.log('Parameters without model:', parametersWithoutModel);
        } else {
          // Pretrained model일 때: 기존 방식대로 model 포함
          requestBody.parameters = yoloParameters;
          console.log('Using pretrained model:', yoloParameters.model);
        }
        
        console.log('=== Final Request Body ===');
        console.log('requestBody:', JSON.stringify(requestBody, null, 2));
        console.log('requestBody.did:', requestBody.did);
        console.log('requestBody.pid:', requestBody.pid);
        console.log('requestBody.user_classes:', requestBody.user_classes);
        console.log('requestBody.parameters:', requestBody.parameters);
        console.log('=== Epoch Debug ===');
        console.log('trainingConfig.algoParams.epochs:', trainingConfig.algoParams.epochs);
        console.log('yoloParameters.epochs:', yoloParameters.epochs);

        const apiRequestData = { uid, ...requestBody };
        console.log('=== API Request Data ===');
        console.log('apiRequestData:', JSON.stringify(apiRequestData, null, 2));
        
        const response = await postYoloTraining(apiRequestData);
        
        // Log training start and response details
        if (response?.message) {
          progress.addLog(response.message);
        } else {
          progress.addLog('Training started successfully.');
        }
        
        // Log response classes for debugging
        console.log('=== API Response Classes ===');
        console.log('Response user_classes:', response?.data?.user_classes);
        console.log('Response model_classes:', response?.data?.model_classes);
        console.log('Response dataset_classes:', response?.data?.dataset_classes);
        
        setTrainingResponse(response);
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
  }, [trainingConfig, progress]);

  return {
    ...progress,
    runTraining,
    trainingResponse
  };
}; 