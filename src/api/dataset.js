
import { fetchAllDatasets } from './datasets.js';

export async function fetchDatasetList() {
  try {
    const result = await fetchAllDatasets();
    return result.data.raw; // Return raw datasets for backward compatibility
  } catch (err) {
    throw new Error('Failed to fetch dataset list');
  }
} 