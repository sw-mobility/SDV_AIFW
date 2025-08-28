import { uid } from './uid.js';

// API 기본 URL 설정
export const API_BASE_URL = 'http://localhost:5002';

export const initializeApp = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/`, {
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


