import React from 'react';
import { ChevronDown } from 'lucide-react';
import styles from './Select.module.css';

export default function Select({
                                   label,
                                   options = [],
                                   value,
                                   onChange,
                                   disabled = false,
                                   error = false,
                                   size = 'medium',
                                   className = '',
                                   required = false,
                                   helperText = '',
                                   placeholder = '',
                                   id,
                                   ...props
                               }) {
    const containerClass = [
        styles.selectContainer,
        styles[size],
        error && styles.error,
        className
    ].filter(Boolean).join(' ');

    const selectClass = [
        styles.selectField,
        error && styles.selectError,
        disabled && styles.disabled
    ].filter(Boolean).join(' ');

    const helperClass = [
        styles.helperText,
        error && styles.helperError
    ].filter(Boolean).join(' ');

    return (
        <div className={containerClass}>
            {label && (
                <label htmlFor={id} className={styles.selectLabel}>
                    {label}
                    {required && <span className={styles.required}>*</span>}
                </label>
            )}

            <div className={styles.selectWrapper}>
                <select
                    id={id}
                    value={value}
                    onChange={onChange}
                    disabled={disabled}
                    className={selectClass}
                    {...props}
                >
                    {placeholder && (
                        <option value="" disabled>{placeholder}</option>
                    )}
                    {options.map((opt, idx) => (
                        <option key={opt.value || idx} value={opt.value}>
                            {opt.label}
                        </option>
                    ))}
                </select>

                <div className={styles.selectArrow}>
                    <ChevronDown size={12} strokeWidth={1.5} />
                </div>
            </div>

            {helperText && <div className={helperClass}>{helperText}</div>}
        </div>
    );
}
