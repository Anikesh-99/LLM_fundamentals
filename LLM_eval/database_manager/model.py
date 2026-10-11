from typing import Optional
from sqlalchemy import String, ForeignKey, Integer, create_engine, JSON
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, relationship
import os

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test.db")
engine = create_engine(f"sqlite:///{_DB_PATH}", echo=True)
Base = declarative_base()

class Testcase(Base):
    __tablename__ = "testcases"

    id: Mapped[int] = mapped_column(primary_key = True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id"))
    query: Mapped[String] = mapped_column(String(1000), nullable=False)
    golden_answer: Mapped[String] = mapped_column(JSON)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[String] = mapped_column(String(15), nullable=False)

class Project(Base):
    __tablename__ = "projects"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[String] = mapped_column(String(100), nullable=False)
    type: Mapped[String] = mapped_column(String(10), nullable=False)

Base.metadata.create_all(engine)