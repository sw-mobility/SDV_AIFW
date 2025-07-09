import React from 'react';
import styles from './Input.module.css';

export default function Input({
                                  label,
                                  type = 'text',
                                  value,
                                  onChange,
                                  placeholder = '',
                                  disabled = false,
                                  error = false,
                                  size = 'medium',
                                  className = '',
                                  required = false,
                                  helperText = '',
                                  ...props
                              }) {
    const containerClass = `${styles.inputContainer} ${styles[size]} ${error ? styles.error : ''} ${className}`;
    const inputClass = `${styles.inputField} ${error ? styles.inputError : ''} ${disabled ? styles.disabled : ''}`;

    return (
        <div className={containerClass}>
            {label && (
                <label className={styles.inputLabel}>
                    {label}
                    {required && <span className={styles.required}>*</span>}
                </label>
            )}
            <div className={styles.inputWrapper}>
                <input
                    type={type}
                    value={value}
                    onChange={onChange}
                    placeholder={placeholder}
                    disabled={disabled}
                    className={inputClass}
                    {...props}
                />
                {error && <div className={styles.errorIcon}>⚠</div>}
            </div>
            {helperText && (
                <div className={`${styles.helperText} ${error ? styles.helperError : ''}`}>
                    {helperText}
                </div>
            )}
        </div>
    );
}
