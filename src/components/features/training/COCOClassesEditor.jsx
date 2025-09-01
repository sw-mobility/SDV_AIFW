import React, { useState, useEffect } from 'react';
import { FileText, RotateCcw } from 'lucide-react';
import styles from './COCOClassesEditor.module.css';

const DEFAULT_COCO_CLASSES = `names:
  0: person
  1: bicycle
  2: car`;

const COCOClassesEditor = ({ value, onChange }) => {
  const [yamlContent, setYamlContent] = useState(value || DEFAULT_COCO_CLASSES);

  useEffect(() => {
    setYamlContent(value || DEFAULT_COCO_CLASSES);
  }, [value]);

  const handleContentChange = (e) => {
    const newContent = e.target.value;
    setYamlContent(newContent);
    onChange?.(newContent);
  };

  const handleReset = () => {
    setYamlContent(DEFAULT_COCO_CLASSES);
    onChange?.(DEFAULT_COCO_CLASSES);
  };



  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.titleSection}>
          <FileText size={18} className={styles.icon} />
          <span className={styles.title}>COCO Classes Configuration</span>
        </div>
                 <div className={styles.actions}>
           <button
             type="button"
             onClick={handleReset}
             className={styles.resetButton}
             title="Reset to default"
           >
             <RotateCcw size={16} />
           </button>
         </div>
      </div>
      
      {/* 에러 메시지 제거 - 사용자가 자유롭게 편집할 수 있도록 */}
      
      <div className={styles.editorContainer}>
        <textarea
          value={yamlContent}
          onChange={handleContentChange}
          className={styles.yamlEditor}
          placeholder="Enter YAML configuration..."
          rows={20}
        />
      </div>
      
      <div className={styles.helpText}>
        <p>• 클래스 번호와 이름을 YAML 형식으로 입력하세요</p>
        <p>• 예시: <code>0: person</code>, <code>1: car</code></p>
        <p>• 클래스 번호는 0부터 시작하는 정수여야 합니다</p>
      </div>
    </div>
  );
};

export default COCOClassesEditor;
