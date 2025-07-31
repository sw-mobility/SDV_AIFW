// Mock data for snapshots
export const mockSnapshots = [
  { id: 'default', name: 'Default Snapshot', description: 'System default snapshot' },
  { id: 'snap1', name: 'MyTrainerSnapshot1', description: 'Default trainer snapshot' },
  { id: 'snap2', name: 'MyTrainerSnapshot2', description: 'Experimental snapshot' },
];

// Mock snapshot files
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

// Mock training execution
export const executeTraining = async (trainingConfig) => {
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