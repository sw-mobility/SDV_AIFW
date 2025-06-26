from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class CategoryPath:
    """카테고리 경로를 관리하는 유틸리티 클래스"""
    @staticmethod
    def join(*parts: str) -> str:
        """카테고리 경로를 생성합니다."""
        return '/'.join(part.strip('/') for part in parts if part)
    
    @staticmethod
    def split(path: str) -> List[str]:
        """카테고리 경로를 분리합니다."""
        return [p for p in path.split('/') if p]
    
    @staticmethod
    def parent(path: str) -> Optional[str]:
        """상위 카테고리 경로를 반환합니다."""
        parts = CategoryPath.split(path)
        return '/'.join(parts[:-1]) if len(parts) > 1 else None
    
    @staticmethod
    def name(path: str) -> str:
        """카테고리 이름을 반환합니다."""
        parts = CategoryPath.split(path)
        return parts[-1] if parts else ''

class BaseCategory(BaseModel):
    """기본 카테고리 모델"""
    path: str  # 전체 경로 (예: "동물/포유류/고양이과")
    name: str  # 카테고리 이름 (예: "고양이과")
    description: Optional[str] = None
    parent_path: Optional[str] = None  # 상위 카테고리 경로
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    image_count: int = 0

    def __init__(self, **data):
        # path가 주어진 경우 name과 parent_path를 자동으로 설정
        if 'path' in data:
            path = data['path']
            if 'name' not in data:
                data['name'] = CategoryPath.name(path)
            if 'parent_path' not in data:
                data['parent_path'] = CategoryPath.parent(path)
        super().__init__(**data)

    def add_subcategory(self, name: str, description: Optional[str] = None) -> 'BaseCategory':
        """하위 카테고리를 생성합니다."""
        subcategory_path = CategoryPath.join(self.path, name)
        return self.__class__(
            path=subcategory_path,
            description=description
        )

    def is_root(self) -> bool:
        """최상위 카테고리인지 확인합니다."""
        return self.parent_path is None
    
    def is_subcategory_of(self, other: 'BaseCategory') -> bool:
        """다른 카테고리의 하위 카테고리인지 확인합니다."""
        return self.path.startswith(f"{other.path}/")

class BaseDataset(BaseModel):
    """기본 데이터셋 모델"""
    name: str
    description: Optional[str] = None
    categories: List[BaseCategory] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    total_images: int = 0

    def add_category(self, path: str, description: Optional[str] = None) -> BaseCategory:
        """새로운 카테고리를 추가합니다."""
        # 중복 확인
        if any(cat.path == path for cat in self.categories):
            raise ValueError(f"Category already exists: {path}")
            
        # 상위 카테고리가 있는 경우, 존재 여부 확인
        parent_path = CategoryPath.parent(path)
        if parent_path and not any(cat.path == parent_path for cat in self.categories):
            raise ValueError(f"Parent category not found: {parent_path}")
            
        category = BaseCategory(path=path, description=description)
        self.categories.append(category)
        self.updated_at = datetime.now()
        return category

    def get_category(self, path: str) -> Optional[BaseCategory]:
        """경로로 카테고리를 찾습니다."""
        return next((cat for cat in self.categories if cat.path == path), None)

    def get_subcategories(self, parent_path: Optional[str] = None) -> List[BaseCategory]:
        """특정 카테고리의 하위 카테고리 목록을 반환합니다."""
        if parent_path is None:
            # 최상위 카테고리들 반환
            return [cat for cat in self.categories if cat.is_root()]
        else:
            # 특정 카테고리의 직계 하위 카테고리들 반환
            return [
                cat for cat in self.categories 
                if cat.parent_path == parent_path
            ]