"""Labeled Images 데이터 모델 패키지"""

from .mongodb import LabeledImageFile, Annotation

__all__ = [
    'LabeledImageFile', 'Annotation'
]
