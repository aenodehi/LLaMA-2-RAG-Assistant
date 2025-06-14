from abc import ABC
import uuid
from typing import Optional
from datetime import datetime
from pydantic import UUID4, BaseModel, Field

from .base import VectorBaseDocument
from .types import DataCategory


class CleanedDocument(VectorBaseDocument, ABC):
    content: str
    platform: str
    author_id: UUID4
    author_full_name: str


class LocalFileDocument(VectorBaseDocument):
    content: str
    file_path: str
    platform: str = "local_filesystem"
    author_id: UUID4 = Field(default_factory=lambda: uuid.UUID(int=0))
    author_full_name: str = "System Generated"

    class Config:
        name = "local_files"
        category = DataCategory.FILES
        use_vector_index = False


class CleanedPostDocument(CleanedDocument):
    image: Optional[str] = None

    class Config:
        name = "cleaned_posts"
        category = DataCategory.POSTS
        use_vector_index = False


class CleanedArticleDocument(CleanedDocument):
    link: str

    class Config:
        name = "cleaned_articles"
        category = DataCategory.ARTICLES
        use_vector_index = False


class CleanedRepositoryDocument(CleanedDocument):
    name: str
    link: str

    class Config:
        name = "cleaned_repositories"
        category = DataCategory.REPOSITORIES
        use_vector_index = False
