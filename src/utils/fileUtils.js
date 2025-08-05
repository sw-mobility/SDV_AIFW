/**
 * 파일 관련 유틸리티 함수들
 */

/**
 * 파일명에서 경로 제거하고 파일명만 반환
 * @param {string} fullPath - 전체 파일 경로
 * @returns {string} 파일명만
 */
export const getFileNameOnly = (fullPath) => {
    if (!fullPath) return 'N/A';
    // 경로 구분자로 분할하여 마지막 부분(파일명)만 반환
    return fullPath.split('/').pop() || fullPath.split('\\').pop() || fullPath;
};

/**
 * 파일명에서 확장자 추출
 * @param {string} fileName - 파일명
 * @returns {string} 확장자 (소문자)
 */
export const getFileExtension = (fileName) => {
    if (!fileName) return 'N/A';
    const extension = fileName.split('.').pop()?.toLowerCase();
    return extension || 'N/A';
};

/**
 * 파일 크기를 읽기 쉬운 형태로 포맷팅
 * @param {number} bytes - 바이트 단위 크기
 * @returns {string} 포맷팅된 크기 (예: "1.5 MB")
 */
export const formatFileSize = (bytes) => {
    if (!bytes) return 'N/A';
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
};

/**
 * 파일 타입을 MIME 타입 기반으로 분류
 * @param {string} fileName - 파일명
 * @returns {string} 파일 타입 카테고리
 */
export const getFileTypeCategory = (fileName) => {
    if (!fileName) return 'Unknown';
    
    const extension = getFileExtension(fileName);
    const imageTypes = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'];
    const documentTypes = ['pdf', 'doc', 'docx', 'txt', 'rtf'];
    const videoTypes = ['mp4', 'avi', 'mov', 'wmv', 'flv'];
    const audioTypes = ['mp3', 'wav', 'flac', 'aac'];
    
    if (imageTypes.includes(extension)) return 'Image';
    if (documentTypes.includes(extension)) return 'Document';
    if (videoTypes.includes(extension)) return 'Video';
    if (audioTypes.includes(extension)) return 'Audio';
    
    return 'Other';
}; 