from typing import Optional
from storage.clients.minio import MinioStorageClient

"""MinIO Storage Client Singleton Manager"""

class StorageInitError(Exception):
    """스토리지 초기화 과정에서 발생하는 예외"""
    pass

class StorageManager:
    """MinIO 스토리지 클라이언트 관리자
    
    싱글톤 패턴을 사용하여 하나의 MinIO 클라이언트 인스턴스만 생성하고 관리합니다.
    """
    _instance: Optional[MinioStorageClient] = None

    @classmethod
    def get_instance(cls) -> MinioStorageClient:
        """MinIO 클라이언트 인스턴스를 반환합니다.
        
        만약 인스턴스가 아직 생성되지 않았다면, 새 인스턴스를 생성합니다.
        
        Returns:
            MinioStorageClient: 초기화된 MinIO 클라이언트
        """
        if cls._instance is None:
            try:
                cls._instance = MinioStorageClient()
            except Exception as e:
                raise StorageInitError(f"Failed to initialize MinIO client: {str(e)}")
        return cls._instance

# Global instance
storage_client = StorageManager.get_instance()
