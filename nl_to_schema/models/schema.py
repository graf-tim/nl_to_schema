from pydantic import BaseModel, Field
from enum import Enum


class Type(str, Enum):
    INTEGER = "INTEGER"
    VARCHAR = "VARCHAR"
    TEXT = "TEXT"
    DATE = "DATE"
    BOOLEAN = "BOOLEAN"
    DECIMAL = "DECIMAL"
    TIMESTAMP = "TIMESTAMP"


class Column(BaseModel):
    name: str = Field(description="Column name in snake_case")
    type: Type = Field(
        description=(
            "SQL data type. Use only one of: INTEGER, VARCHAR, "
            "TEXT, DATE, BOOLEAN, DECIMAL, TIMESTAMP"
        )
    )
    primary_key: bool = Field(
        default=False,
        description="True if this column is part of the primary key"
    )
    nullable: bool = Field(
        default=True,
        description="True if this column may contain NULL values"
    )


class ForeignKey(BaseModel):
    from_column: str = Field(description="Name of the foreign key column in this table")
    references_table: str = Field(description="Name of the referenced table")
    references_column: str = Field(description="Name of the referenced column")


class Table(BaseModel):
    name: str = Field(description="Table name in snake_case")
    columns: list[Column] = Field(description="List of all columns in the table")
    foreign_keys: list[ForeignKey] = Field(default_factory=list, description="List of all foreign key relationships")


class LogicalSchema(BaseModel):
    tables: list[Table] = Field(description="List of all tables in the relational schema")
