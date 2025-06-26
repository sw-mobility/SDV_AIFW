"""파일 해시 계산 유틸리티"""
import hashlib
from typing import Union, Optional
from io import BytesIO

def calculate_file_hash(content: Union[bytes, BytesIO], algorithm: str = 'sha256') -> str:
    """
    파일 컨텐츠의 해시 값을 계산합니다.
    
    Args:
        content (bytes | BytesIO): 해시 값을 계산할 바이트 데이터 또는 BytesIO 객체
        algorithm (str): 해시 알고리즘 ('md5', 'sha1', 'sha256', 'sha512')
        
    Returns:
        str: 계산된 해시 값 (16진수 문자열)
    """
    if algorithm.lower() == 'md5':
        hash_obj = hashlib.md5()
    elif algorithm.lower() == 'sha1':
        hash_obj = hashlib.sha1()
    elif algorithm.lower() == 'sha256':
        hash_obj = hashlib.sha256()
    elif algorithm.lower() == 'sha512':
        hash_obj = hashlib.sha512()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    # BytesIO 객체인 경우 처리
    if isinstance(content, BytesIO):
        # 현재 위치 저장
        current_position = content.tell()
        # 처음 위치로 이동
        content.seek(0)
        # 청크 단위로 읽어서 해시 계산
        chunk_size = 65536  # 64KB 청크
        data = content.read(chunk_size)
        while data:
            hash_obj.update(data)
            data = content.read(chunk_size)
        # 원래 위치로 복원
        content.seek(current_position)
    # bytes인 경우 처리
    elif isinstance(content, bytes):
        hash_obj.update(content)
    else:
        raise TypeError("Content must be bytes or BytesIO object")
    
    # 16진수 문자열로 반환
    return hash_obj.hexdigest()
