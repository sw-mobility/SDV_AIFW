import React, { useState, useEffect } from 'react';
import { FileText, RotateCcw } from 'lucide-react';
import styles from './COCOClassesEditor.module.css';

const DEFAULT_COCO_CLASSES = `  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush`;

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
