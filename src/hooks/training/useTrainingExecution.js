import { useCallback, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { validateTrainingExecution } from '../../domain/training/trainingValidation.js';
import { useProgress } from '../common/useProgress.js';
import { postYoloTraining, getYoloDefaultYaml } from '../../api/training.js';
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
    const validation = validateTrainingExecution(trainingConfig);
    
    if (!validation.isValid) {
      const errorMessages = validation.errors.map(error => error.message);
      alert(errorMessages.join('\n'));
      return;
    }

    if (!projectData) {
      progress.addLog('Error: Project data not available');
      return;
    }

    if (!trainingConfig.selectedDataset) {
      progress.addLog('Error: No dataset selected');
      return;
    }

    progress.start();
    progress.addLog('Training started...');

    try {
      if (trainingConfig.algorithm === 'YOLO' || trainingConfig.algorithm === 'yolo_v8') {
        progress.addLog('Starting YOLO training...');
        

        
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
          
          // Temporary: Set default classes if none provided
          const defaultClasses = ['person', 'bicycle', 'car'];
          console.log('Using default classes:', defaultClasses);
          userClasses = defaultClasses;
        }
        
        // Temporary: Force test classes to verify the flow
        if (userClasses.length === 0) {
          userClasses = ['test_class_1', 'test_class_2', 'test_class_3'];
          console.log('Forcing test classes:', userClasses);
        }
        
        console.log('Final userClasses after parsing:', userClasses);

        // Prepare YOLO parameters based on API schema
        const yoloParameters = {
          model: trainingConfig.algoParams.model || 'yolov8n',
          split_ratio: splitRatio,
          epochs: trainingConfig.algoParams.epochs || 5,
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
          resume: trainingConfig.algoParams.resume === true,
          amp: trainingConfig.algoParams.amp !== false,
          single_cls: trainingConfig.algoParams.single_cls !== false,
          cos_lr: trainingConfig.algoParams.cos_lr !== false,
          close_mosaic: trainingConfig.algoParams.close_mosaic || 0,
          overlap_mask: trainingConfig.algoParams.overlap_mask !== false
        };

        // Get project and dataset IDs
        const projectId = projectData.pid || projectData._id || projectData.id;
        const datasetId = trainingConfig.selectedDataset.did;
        const datasetClasses = trainingConfig.selectedDataset.classes || [];
        
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
          // Use default classes instead of dataset classes
          finalUserClasses = ['person', 'bicycle', 'car'];
          console.log('Using default classes instead of dataset classes:', finalUserClasses);
        }
        
        // Log which classes are being used
        if (userClasses.length > 0) {
          console.log('Using user-defined classes:', userClasses);
        } else {
          console.log('No user classes provided, using dataset classes:', datasetClassesList);
        }
        
        console.log('=== Classes Debug ===');
        console.log('User input classes:', userClasses);
        console.log('Dataset classes:', datasetClassesList);
        console.log('Final classes being sent:', finalUserClasses);
        console.log('Request body classes:', {
          user_classes: finalUserClasses,
          model_classes: finalUserClasses
        });
        
        const requestBody = {
          pid: projectId,
          did: datasetId,
          user_classes: finalUserClasses,
          parameters: yoloParameters
        };
        
        console.log('Final request body classes:', {
          user_classes: finalUserClasses
        });

        const response = await postYoloTraining({ uid, ...requestBody });
        
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
  }, [trainingConfig, progress, projectData, projectName]);

  return {
    ...progress,
    runTraining,
    trainingResponse
  };
}; 