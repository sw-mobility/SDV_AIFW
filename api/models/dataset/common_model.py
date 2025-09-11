from fastapi import Form
from typing import List, Optional
from pydantic import BaseModel, Field

class DatasetDelete(BaseModel):
    target_did_list: List[str]
    target_path_list: List[str]

class DataDelete(BaseModel):
    target_did: str
    target_name_list: List[str]
    target_path_list: List[str]

class DatasetDownload(BaseModel):
    target_did: str

class DataDownload(BaseModel):
    target_path_list: List[str]