"""원본 데이터셋 저장소 관리자"""
from . import images  # 이미지 데이터셋 저장소 관리자

# 편의성을 위한 직접 인스턴스 가져오기
from .images import RawImageStorageManager

__all__ = ['images', 'RawImageStorageManager']