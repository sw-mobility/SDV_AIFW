import React from 'react';
import './Select.module.css';

export default function Select({
                                   label,
                                   options = [],
                                   value,
                                   onChange,
                                   disabled = false,
                                   className = '',
                                   ...props
                               }) {
    return (
        <div className={`select-container ${className}`}>
            {label && <label className="select-label">{label}</label>}
            <select
                value={value}
                onChange={onChange}
                disabled={disabled}
                className="select-field"
                {...props}
            >
                {options.map((opt, idx) => (
                    <option key={idx} value={opt.value}>
                        {opt.label}
                    </option>
                ))}
            </select>
        </div>
    );
}
