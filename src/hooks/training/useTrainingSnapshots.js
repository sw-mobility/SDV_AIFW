import { useState, useEffect } from 'react';
import { mockSnapshots, snapshotFiles } from '../../mocks/trainingSnapshots.js';

export const useTrainingSnapshots = () => {
  const [snapshots, setSnapshots] = useState(mockSnapshots);
  const [selectedSnapshot, setSelectedSnapshot] = useState(null);
  const [editorFileStructure, setEditorFileStructure] = useState(snapshotFiles.default.fileStructure);
  const [editorFiles, setEditorFiles] = useState(snapshotFiles.default.files);

  useEffect(() => {
    const snapId = selectedSnapshot ? selectedSnapshot.id : 'default';
    setEditorFileStructure(snapshotFiles[snapId]?.fileStructure || snapshotFiles.default.fileStructure);
    setEditorFiles(snapshotFiles[snapId]?.files || snapshotFiles.default.files);
  }, [selectedSnapshot]);

  const selectSnapshot = (snapshot) => {
    setSelectedSnapshot(snapshot);
  };

  const clearSelectedSnapshot = () => {
    setSelectedSnapshot(null);
  };

  const updateSnapshots = (newSnapshots) => {
    setSnapshots(newSnapshots);
  };

  return {
    snapshots,
    selectedSnapshot,
    setSelectedSnapshot: selectSnapshot,
    clearSelectedSnapshot,
    editorFileStructure,
    editorFiles,
    updateSnapshots
  };
}; 