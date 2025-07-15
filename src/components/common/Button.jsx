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
const Button = ({
                    children,
                    variant = 'primary',
                    size = 'medium',
                    disabled = false,
                    onClick,
                    type = 'button',
                    className = '',
                    ...props
                }) => {
    const buttonClasses = [
        styles.button,
        styles[variant],
        styles[size],
        disabled ? styles.disabled : '',
        className
    ].filter(Boolean).join(' '); //배열에서 falsy 값 제거, 배열 요소 공백으로 이어붙임

    return (
        <button
            type={type}
            className={buttonClasses}
            onClick={onClick}
            disabled={disabled}
            {...props}
        >
            {children}
        </button>
    );
};

export default Button;
