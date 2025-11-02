"""
SQLAlchemy ORM models for PDF research paper indexing database.

This module defines the database schema using SQLAlchemy's declarative base.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, Text, DateTime, ForeignKey, Float, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from typing import List


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Paper(Base):
    """
    Represents a research paper in the database.
    
    Attributes:
        paper_id: Primary key (auto-incremented integer)
        url: Original download URL of the paper
        filename: Filename in the ./papers/ directory
        download_date: Timestamp when the paper was downloaded
        title: Extracted title of the paper (optional)
        num_chunks: Total number of chunks for this paper
        num_pages: Total number of pages in the PDF
        chunking_method: Method used to chunk the paper (header or s2)
        chunks: Relationship to Chunk objects
        sections: Relationship to Section objects
    """
    __tablename__ = "papers"
    
    paper_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    url: Mapped[str] = mapped_column(String(1000), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    download_date: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    title: Mapped[Optional[str]] = mapped_column(String(500))
    num_chunks: Mapped[int] = mapped_column(Integer, default=0)
    num_pages: Mapped[int] = mapped_column(Integer, default=0)
    chunking_method: Mapped[Optional[str]] = mapped_column(String(20))
    
    # Relationships
    chunks: Mapped[List["Chunk"]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    sections: Mapped[List["Section"]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"Paper(paper_id={self.paper_id!r}, filename={self.filename!r}, num_chunks={self.num_chunks!r})"


class Chunk(Base):
    """
    Represents a text chunk from a paper.
    
    Attributes:
        chunk_id: Primary key (auto-incremented integer)
        paper_id: Foreign key to Paper
        chunk_index: Sequential index within the paper (0-based)
        text: Full text content of the chunk
        header_path: Hierarchical header path (e.g., "Introduction > Background")
        header_level: Header level (1-3 for headers, 0 for content)
        page_start: First page number (0-indexed)
        page_end: Last page number (0-indexed)
        prev_chunk_index: Index of previous chunk (None if first)
        next_chunk_index: Index of next chunk (None if last)
        section_id: Foreign key to Section (optional)
        embedding_index: Index in FAISS vector store (optional)
        paper: Relationship to Paper object
        section: Relationship to Section object
    """
    __tablename__ = "chunks"
    
    chunk_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.paper_id"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    header_path: Mapped[Optional[str]] = mapped_column(String(1000))
    header_level: Mapped[int] = mapped_column(Integer, default=0)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    prev_chunk_index: Mapped[Optional[int]] = mapped_column(Integer)
    next_chunk_index: Mapped[Optional[int]] = mapped_column(Integer)
    section_id: Mapped[Optional[int]] = mapped_column(ForeignKey("sections.section_id"))
    embedding_index: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Relationships
    paper: Mapped["Paper"] = relationship(back_populates="chunks")
    section: Mapped[Optional["Section"]] = relationship(back_populates="chunks")
    
    def __repr__(self) -> str:
        return f"Chunk(chunk_id={self.chunk_id!r}, paper_id={self.paper_id!r}, chunk_index={self.chunk_index!r})"


class Section(Base):
    """
    Represents a document section (based on header paths).
    
    Attributes:
        section_id: Primary key (auto-incremented integer)
        paper_id: Foreign key to Paper
        header_path: Full hierarchical header path
        header_level: Level of the header (1-3)
        start_chunk_index: First chunk index in this section
        end_chunk_index: Last chunk index in this section
        page_start: First page of the section
        page_end: Last page of the section
        paper: Relationship to Paper object
        chunks: Relationship to Chunk objects
    """
    __tablename__ = "sections"
    
    section_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.paper_id"), nullable=False)
    header_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    header_level: Mapped[int] = mapped_column(Integer, nullable=False)
    start_chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    end_chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Relationships
    paper: Mapped["Paper"] = relationship(back_populates="sections")
    chunks: Mapped[List["Chunk"]] = relationship(back_populates="section")
    
    def __repr__(self) -> str:
        return f"Section(section_id={self.section_id!r}, header_path={self.header_path!r})"

