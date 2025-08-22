import { uid } from './uid.js';

export const initializeApp = async () => {
  try {
    const response = await fetch('/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'uid': uid
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    console.log('App initialized successfully:', data);
    return data;
  } catch (error) {
    console.error('Failed to initialize app:', error);
    throw error;
  }
};

