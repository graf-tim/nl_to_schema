"""Pydantic-Modelle für die Testfall-Generierung.

LogicalSchema/Column/Table sind strukturell identisch mit dem Evaluations-
Rahmen unter nl_to_schema/, werden hier aber bewusst dupliziert, damit
testdata_generator/ als eigenständiges Tool ohne Importpfad-Kopplung lauffähig
ist.

"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

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


class DesignBegruendung(BaseModel):
    entitaetsentscheidungen: str = Field(description="Decisions about entity definitions")
    beziehungsentscheidungen: str = Field(description="Decisions about relationship definitions")
    normalisierungshinweise: str = Field(description="Notes about normalization")
    ambiguitaeten: str = Field(description="Ambiguities in the design")


class Testfall(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    stufe: int
    domaene: str = Field(description="Domain of the test case")
    anforderungstext: str = Field(description="Requirement text for the test case")
    referenzschema: LogicalSchema = Field(description="Reference schema for the test case")
    begruendung: DesignBegruendung = Field(description="Design justification for the test case")
    generiert_mit: str = Field(default="", description="Tool used to generate the test case")
    generiert_am: str = Field(default="", description="Date when the test case was generated")
    prompt_version: str = Field(default="", description="Version of the prompt used for generation")