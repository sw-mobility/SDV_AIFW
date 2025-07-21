
import { fetchAllDatasets } from './datasets.js';

// fetchDatasetList는 더 이상 사용하지 않음. context에서 fetchAllDatasets를 직접 사용.
export async function fetchDatasetList() {
  try {
    const result = await fetchAllDatasets();
    // id, path 등 불필요한 필드 제거
    return result.data.raw.map(ds => {
      const {
        name, description, type, created_at, status, task_type, label_format, total, origin_raw
      } = ds;
      return {
        name, description, type, created_at, status, task_type, label_format, total, origin_raw
      };
    });
  } catch (err) {
    throw new Error('Failed to fetch dataset list');
  }
} 