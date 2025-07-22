import React from 'react';
import styles from './Button.module.css';

/**
 * 버튼 컴포넌트 사용 방법 예시
 *
 * <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '10px' }}>
 *           <Button variant="primary">Primary</Button>
 *           <Button variant="secondary">Secondary</Button>
 *           <Button variant="outline">Outline</Button>
 *           <Button variant="ghost">Ghost</Button>
 *           <Button variant="danger">Danger</Button>
 * </div>
 */
export default function Button({ children, variant = 'primary', size = 'medium', onClick, disabled, style, icon }) {
  let className = styles.button;
  if (variant === 'primary' || variant === 'primary-gradient') className += ' ' + styles.primary;
  if (variant === 'secondary') className += ' ' + styles.secondary;
  if (variant === 'danger') className += ' ' + styles.danger;
  if (size === 'large') className += ' ' + styles.large;
  if (size === 'medium') className += ' ' + styles.medium;
  if (disabled) className += ' ' + styles.disabled;
  return (
    <button className={className} onClick={onClick} disabled={disabled} style={style}>
      {icon && <span className={styles.iconWrap}>{icon}</span>}
      {children}
    </button>
  );
}
