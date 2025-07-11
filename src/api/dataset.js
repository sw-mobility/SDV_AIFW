export async function fetchDatasetList() {
  try {
    // 실제 API 엔드포인트로 교체 필요
    // const response = await axios.get('/api/datasets');
    // return response.data;
    // 임시 mock 데이터
    return [
      { id: 1, name: 'Image Dataset 1', type: 'Image', size: '2.3GB', lastModified: '2024-01-15', status: 'Active' },
      { id: 2, name: 'Text Dataset 1', type: 'Text', size: '1.5GB', lastModified: '2024-01-14', status: 'Active' },
      { id: 3, name: 'Audio Dataset 1', type: 'Audio', size: '3.2GB', lastModified: '2024-01-13', status: 'Active' },
    ];
  } catch (err) {
    throw new Error('Failed to fetch dataset list');
  }
} 