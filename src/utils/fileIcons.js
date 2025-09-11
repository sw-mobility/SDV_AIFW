/**
 * VSCode 스타일 파일 아이콘 매핑
 * @iconify/icons-vscode-icons 패키지의 아이콘들을 사용
 */

// 파일 확장자별 아이콘 매핑
const fileIconMap = {
  // Python
  'py': 'vscode-icons:file-type-python',
  'pyc': 'vscode-icons:file-type-python',
  
  // JSON/YAML/Config
  'json': 'vscode-icons:file-type-json',
  'yaml': 'vscode-icons:file-type-yaml',
  'yml': 'vscode-icons:file-type-yaml',
  'toml': 'vscode-icons:file-type-toml',
  'cfg': 'vscode-icons:file-type-config',
  'conf': 'vscode-icons:file-type-config',
  'ini': 'vscode-icons:file-type-config',
  
  // Markup/Documentation
  'md': 'vscode-icons:file-type-markdown',
  'html': 'vscode-icons:file-type-html',
  'htm': 'vscode-icons:file-type-html',
  'xml': 'vscode-icons:file-type-xml',
  'txt': 'vscode-icons:file-type-text',
  'log': 'vscode-icons:file-type-log',
  // Shell/Scripts
  'sh': 'vscode-icons:file-type-shell',
  'bash': 'vscode-icons:file-type-shell',
  'zsh': 'vscode-icons:file-type-shell',
  'fish': 'vscode-icons:file-type-shell',
  'bat': 'vscode-icons:file-type-bat',
  'cmd': 'vscode-icons:file-type-bat',
  'ps1': 'vscode-icons:file-type-powershell',
  
  // Images
  'png': 'vscode-icons:file-type-image',
  'jpg': 'vscode-icons:file-type-image',
  'jpeg': 'vscode-icons:file-type-image',
  'gif': 'vscode-icons:file-type-image',
  'bmp': 'vscode-icons:file-type-image',
  'svg': 'vscode-icons:file-type-svg',
  'ico': 'vscode-icons:file-type-image',
  'webp': 'vscode-icons:file-type-image',
  
  // Archives
  'zip': 'vscode-icons:file-type-zip',
  'tar': 'vscode-icons:file-type-zip',
  'gz': 'vscode-icons:file-type-zip',
  'rar': 'vscode-icons:file-type-zip',
  '7z': 'vscode-icons:file-type-zip',
  
  // Others
  'pdf': 'vscode-icons:file-type-pdf2',
  'doc': 'vscode-icons:file-type-word',
  'docx': 'vscode-icons:file-type-word',
  'xls': 'vscode-icons:file-type-excel',
  'xlsx': 'vscode-icons:file-type-excel',
  'ppt': 'vscode-icons:file-type-powerpoint',
  'pptx': 'vscode-icons:file-type-powerpoint',
};

// 특별한 파일명에 대한 아이콘 매핑
const specialFileMap = {
  // Python specific
  '__init__.py': 'vscode-icons:file-type-python',
  'requirements.txt': 'vscode-icons:file-type-pip',
  'setup.py': 'vscode-icons:file-type-python',
  'manage.py': 'vscode-icons:file-type-django',
  'wsgi.py': 'vscode-icons:file-type-python',
  'asgi.py': 'vscode-icons:file-type-python',
  
  // JavaScript/Node.js specific
  'package.json': 'vscode-icons:file-type-npm',
  'package-lock.json': 'vscode-icons:file-type-npm',
  'yarn.lock': 'vscode-icons:file-type-yarn',
  'webpack.config.js': 'vscode-icons:file-type-webpack',
  'vite.config.js': 'vscode-icons:file-type-vite',
  'rollup.config.js': 'vscode-icons:file-type-rollup',
  'babel.config.js': 'vscode-icons:file-type-babel2',
  'tsconfig.json': 'vscode-icons:file-type-tsconfig',
  // Config files
  '.env': 'vscode-icons:file-type-dotenv'
};

// 기본 폴더 아이콘 상수 (검정색 단색 아이콘)
const DEFAULT_FOLDER_ICON = 'ph:folder';
const DEFAULT_FOLDER_OPEN_ICON = 'ph:folder-open';

/**
 * 파일명에서 확장자 추출
 * @param {string} fileName - 파일명
 * @returns {string} 확장자 (소문자)
 */
const getFileExtension = (fileName) => {
  if (!fileName || typeof fileName !== 'string') return '';
  
  const lastDotIndex = fileName.lastIndexOf('.');
  if (lastDotIndex === -1 || lastDotIndex === 0) return '';
  
  return fileName.slice(lastDotIndex + 1).toLowerCase();
};

/**
 * 파일 아이콘 가져오기
 * @param {string} fileName - 파일명 또는 경로
 * @returns {string} Iconify 아이콘 이름
 */
export const getFileIcon = (fileName) => {
  if (!fileName) return 'vscode-icons:default-file';
  
  // 경로에서 파일명만 추출
  const baseFileName = fileName.split('/').pop() || fileName;
  
  // 특별한 파일명 확인
  if (specialFileMap[baseFileName]) {
    return specialFileMap[baseFileName];
  }
  
  // 확장자로 아이콘 찾기
  const extension = getFileExtension(baseFileName);
  if (extension && fileIconMap[extension]) {
    return fileIconMap[extension];
  }
  
  // 기본 파일 아이콘
  return 'vscode-icons:default-file';
};

/**
 * 폴더 아이콘 가져오기 (닫힌 상태)
 * @param {string} folderName - 폴더명
 * @returns {string} Iconify 아이콘 이름
 */
export const getFolderIcon = () => DEFAULT_FOLDER_ICON;

/**
 * 폴더 아이콘 가져오기 (열린 상태)
 * @param {string} folderName - 폴더명
 * @returns {string} Iconify 아이콘 이름
 */
export const getOpenFolderIcon = () => DEFAULT_FOLDER_OPEN_ICON;

/**
 * 파일 타입 감지 (프로그래밍 언어)
 * @param {string} fileName - 파일명
 * @returns {string} 프로그래밍 언어
 */
export const getFileLanguage = (fileName) => {
  const extension = getFileExtension(fileName);
  
  const languageMap = {
    'py': 'python',
    'pyc': 'python', // 컴파일된 Python 바이트코드
    'js': 'javascript',
    'jsx': 'javascript',
    'ts': 'typescript',
    'tsx': 'typescript',
    'json': 'json',
    'yaml': 'yaml',
    'yml': 'yaml',
    'md': 'markdown',
    'html': 'html',
    'css': 'css',
    'scss': 'scss',
    'sass': 'sass',
    'sh': 'shell',
    'bash': 'shell',
    'txt': 'plaintext',
    'log': 'plaintext',
    'cfg': 'ini',
    'conf': 'ini',
    'ini': 'ini',
  };
  
  return languageMap[extension] || 'plaintext';
};
