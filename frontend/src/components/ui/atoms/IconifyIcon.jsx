import React from 'react';
import { Icon } from '@iconify/react';

/**
 * Iconify 아이콘 컴포넌트
 * @param {string} icon - Iconify 아이콘 이름 (예: 'vscode-icons:file-type-python')
 * @param {number} size - 아이콘 크기 (픽셀)
 * @param {string} color - 아이콘 색상 (CSS 색상 값)
 * @param {string} className - 추가 CSS 클래스
 * @param {object} style - 인라인 스타일
 * @param {...object} props - 기타 props
 * @returns {JSX.Element}
 */
const IconifyIcon = ({ 
  icon, 
  size = 16, 
  color, 
  className = '', 
  style = {}, 
  ...props 
}) => {
  // 기본 스타일 설정
  const iconStyle = {
    width: size,
    height: size,
    flexShrink: 0,
    ...style
  };

  // 색상이 지정된 경우 스타일에 추가
  if (color) {
    iconStyle.color = color;
  }

  return (
    <Icon 
      icon={icon} 
      style={iconStyle}
      className={className}
      {...props}
    />
  );
};

export default IconifyIcon;

