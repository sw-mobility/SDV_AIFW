import { useState, useEffect } from 'react';

export const useLocalStorageState = (key, defaultValue) => {
    const [state, setState] = useState(() => {
        const storedValue = localStorage.getItem(key);
        return storedValue ? JSON.parse(storedValue) : defaultValue;
    });

    useEffect(() => {
        localStorage.setItem(key, JSON.stringify(state));
    }, [key, state]);

    return [state, setState];
}; 