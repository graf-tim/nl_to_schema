"""DDL-Generator: Determinismus, topologische Sortierung, Validierung."""
import pytest

from ddl_generator import generate_ddl, validate_ddl_structural
from models.schema import Column, DataType, ForeignKey, LogicalSchema, Table


def _simple_schema() -> LogicalSchema:
    return LogicalSchema(
        tables=[
            Table(
                name="autor",
                columns=[
                    Column(name="id", data_type=DataType.INTEGER, nullable=False, primary_key=True),
                    Column(name="name", data_type=DataType.VARCHAR, nullable=False),
                ],
            ),
            Table(
                name="buch",
                columns=[
                    Column(name="id", data_type=DataType.INTEGER, nullable=False, primary_key=True),
                    Column(name="titel", data_type=DataType.VARCHAR, nullable=False),
                    Column(name="autor_id", data_type=DataType.INTEGER, nullable=False),
                ],
                foreign_keys=[
                    ForeignKey(from_column="autor_id", references_table="autor", references_column="id"),
                ],
            ),
        ]
    )


def test_generate_ddl_is_deterministic():
    schema = _simple_schema()
    a = generate_ddl(schema)
    b = generate_ddl(schema)
    assert a == b
    assert "CREATE TABLE autor" in a
    assert "CREATE TABLE buch" in a
    assert a.index("CREATE TABLE autor") < a.index("CREATE TABLE buch")


def test_validate_ddl_structural_clean():
    schema = _simple_schema()
    ddl = generate_ddl(schema)
    result = validate_ddl_structural(ddl, schema)
    assert result["syntaktisch_korrekt"] is True
    assert result["pk_vollstaendigkeit"] == 1.0
    assert result["fk_integritaet"] == 1.0


def test_validate_ddl_structural_broken_fk():
    schema = _simple_schema()
    schema.tables[1].foreign_keys[0].references_table = "nichtexistent"
    ddl = generate_ddl(schema)
    result = validate_ddl_structural(ddl, schema)
    assert result["fk_integritaet"] < 1.0


def test_circular_reference_raises():
    schema = LogicalSchema(
        tables=[
            Table(
                name="a",
                columns=[
                    Column(name="id", data_type=DataType.INTEGER, nullable=False, primary_key=True),
                    Column(name="b_id", data_type=DataType.INTEGER, nullable=False),
                ],
                foreign_keys=[ForeignKey(from_column="b_id", references_table="b", references_column="id")],
            ),
            Table(
                name="b",
                columns=[
                    Column(name="id", data_type=DataType.INTEGER, nullable=False, primary_key=True),
                    Column(name="a_id", data_type=DataType.INTEGER, nullable=False),
                ],
                foreign_keys=[ForeignKey(from_column="a_id", references_table="a", references_column="id")],
            ),
        ]
    )
    with pytest.raises(ValueError, match="Zirkelreferenz"):
        generate_ddl(schema)
