from pydantic import BaseModel, Field
from enum import Enum


class DataType(str, Enum):
    INTEGER = "INTEGER"
    VARCHAR = "VARCHAR"
    TEXT = "TEXT"
    DATE = "DATE"
    BOOLEAN = "BOOLEAN"
    DECIMAL = "DECIMAL"
    TIMESTAMP = "TIMESTAMP"


class Column(BaseModel):
    name: str
    data_type: DataType
    nullable: bool = True
    primary_key: bool = False


class ForeignKey(BaseModel):
    from_column: str
    to_table: str
    to_column: str


class Table(BaseModel):
    name: str
    columns: list[Column]
    foreign_keys: list[ForeignKey] = Field(default_factory=list)


class LogicalSchema(BaseModel):
    tables: list[Table]
