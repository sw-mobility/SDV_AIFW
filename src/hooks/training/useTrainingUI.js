import { useState, useCallback } from 'react';

export const useTrainingUI = () => {
  const [openParamGroup, setOpenParamGroup] = useState(0);
  const [showCodeEditor, setShowCodeEditor] = useState(false);
  const [selectedParamKeys, setSelectedParamKeys] = useState([]);

  const toggleParamKey = useCallback((key) => {
    setSelectedParamKeys(prev =>
      prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]
    );
  }, []);

  const removeParamKey = useCallback((key) => {
    setSelectedParamKeys(prev => prev.filter(k => k !== key));
  }, []);

  const resetUI = useCallback(() => {
    setOpenParamGroup(0);
    setShowCodeEditor(false);
    setSelectedParamKeys([]);
  }, []);

  const toggleCodeEditor = useCallback(() => {
    setShowCodeEditor(prev => !prev);
  }, []);

  const openParamGroupHandler = useCallback((groupIndex) => {
    setOpenParamGroup(groupIndex);
  }, []);

  return {
    openParamGroup,
    setOpenParamGroup: openParamGroupHandler,
    showCodeEditor,
    setShowCodeEditor: toggleCodeEditor,
    selectedParamKeys,
    setSelectedParamKeys,
    toggleParamKey,
    removeParamKey,
    resetUI
  };
}; 