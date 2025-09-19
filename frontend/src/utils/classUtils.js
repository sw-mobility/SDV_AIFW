/**
 * Dataset의 classes 배열을 YAML 구조로 변환
 * @param {Array} classes - 클래스 이름 배열 (예: ["Ambulance", "Bus", "Car"])
 * @returns {string} YAML 형식의 문자열 (예: "0: Ambulance\n1: Bus\n2: Car")
 */
export const convertClassesToYaml = (classes) => {
  if (!classes || !Array.isArray(classes)) {
    return '';
  }
  
  return classes
    .map((className, index) => `${index}: ${className}`)
    .join('\n');
};

/**
 * YAML 구조를 classes 배열로 변환
 * @param {string} yamlString - YAML 형식의 문자열 (예: "0: Ambulance\n1: Bus\n2: Car")
 * @returns {Array} 클래스 이름 배열 (예: ["Ambulance", "Bus", "Car"])
 */
export const convertYamlToClasses = (yamlString) => {
  if (!yamlString || typeof yamlString !== 'string') {
    return [];
  }
  
  try {
    const lines = yamlString.trim().split('\n');
    const classes = [];
    
    lines.forEach(line => {
      const match = line.match(/^(\d+):\s*(.+)$/);
      if (match) {
        const index = parseInt(match[1]);
        const className = match[2].trim();
        classes[index] = className;
      }
    });
    
    // undefined 값 제거하고 순서대로 정렬
    return classes.filter(Boolean);
  } catch (error) {
    console.error('Error parsing YAML string:', error);
    return [];
  }
};
