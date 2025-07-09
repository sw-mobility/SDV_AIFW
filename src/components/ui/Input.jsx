import React from 'react';
import './Input.module.css';

export default function Input({
                                  label,
                                  type = 'text',
                                  value,
                                  onChange,
                                  placeholder = '',
                                  disabled = false,
                                  className = '',
                                  ...props
                              }) {
    return (
        <div className={`input-container ${className}`}>
            {label && <label className="input-label">{label}</label>}
            <input
                type={type}
                value={value}
                onChange={onChange}
                placeholder={placeholder}
                disabled={disabled}
                className="input-field"
                {...props}
            />
        </div>
    );
}
