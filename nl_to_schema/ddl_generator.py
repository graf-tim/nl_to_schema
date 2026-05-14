"""Deterministischer DDL-Generator und strukturelle Validierung."""
from __future__ import annotations

import sqlparse

from models.schema import LogicalSchema, Table, Column, ForeignKey, Type


_TYPE_RENDER = {
    Type.INTEGER: "INTEGER",
    Type.VARCHAR: "VARCHAR(255)",
    Type.TEXT: "TEXT",
    Type.DATE: "DATE",
    Type.BOOLEAN: "BOOLEAN",
    Type.DECIMAL: "DECIMAL(18,2)",
    Type.TIMESTAMP: "TIMESTAMP",
}


def _topological_sort(tables: list[Table]) -> list[Table]:
    by_name = {t.name: t for t in tables}
    visited: set[str] = set()
    on_stack: set[str] = set()
    order: list[Table] = []

    def visit(name: str, path: list[str]) -> None:
        if name in visited:
            return
        if name in on_stack:
            cycle = " -> ".join(path + [name])
            raise ValueError(
                f"Zirkelreferenz im Schema entdeckt: {cycle}. "
                f"Topologische Sortierung nicht möglich."
            )
        if name not in by_name:
            return
        on_stack.add(name)
        table = by_name[name]
        for fk in table.foreign_keys:
            if fk.references_table != name and fk.references_table in by_name:
                visit(fk.references_table, path + [name])
        on_stack.discard(name)
        visited.add(name)
        order.append(by_name[name])

    for t in sorted(tables, key=lambda x: x.name):
        visit(t.name, [])
    return order


def _render_column(col: Column) -> str:
    parts = [col.name, _TYPE_RENDER[col.type]]
    if not col.nullable:
        parts.append("NOT NULL")
    return " ".join(parts)


def _render_table(table: Table) -> str:
    lines: list[str] = []
    for col in table.columns:
        lines.append(f"    {_render_column(col)}")

    pk_cols = [c.name for c in table.columns if c.primary_key]
    if pk_cols:
        lines.append(
            f"    CONSTRAINT pk_{table.name} PRIMARY KEY ({', '.join(pk_cols)})"
        )

    for fk in table.foreign_keys:
        lines.append(
            f"    CONSTRAINT fk_{table.name}_{fk.from_column} "
            f"FOREIGN KEY ({fk.from_column}) "
            f"REFERENCES {fk.references_table} ({fk.references_column})"
        )

    body = ",\n".join(lines)
    return f"CREATE TABLE {table.name} (\n{body}\n);"


def generate_ddl(schema: LogicalSchema) -> str:
    """Erzeugt deterministisches DDL aus einem LogicalSchema.

    Reihenfolge: topologisch sortiert (referenzierte Tabellen zuerst).
    """
    if not schema.tables:
        return ""
    sorted_tables = _topological_sort(list(schema.tables))
    return "\n\n".join(_render_table(t) for t in sorted_tables) + "\n"


def _is_syntactically_valid(ddl: str) -> bool:
    """Prüft mit sqlparse, ob das DDL als CREATE-TABLE-Statements parsbar ist."""
    if not ddl.strip():
        return False
    try:
        statements = sqlparse.parse(ddl)
    except Exception:
        return False
    if not statements:
        return False
    for stmt in statements:
        text = str(stmt).strip()
        if not text:
            continue
        if stmt.get_type() != "CREATE":
            return False
        upper = text.upper()
        if "CREATE TABLE" not in upper:
            return False
        if upper.count("(") != upper.count(")"):
            return False
    return True


def validate_ddl_structural(ddl: str, schema: LogicalSchema) -> dict:
    """Strukturelle Validierung als Gate-Kriterium plus PK/FK-Quoten und absolute Counts.

    Felder:
      - syntaktisch_korrekt:        bool (Gate)
      - pk_vollstaendigkeit:        float in [0,1]  (Rate)
      - fk_integritaet:             float in [0,1]  (Rate, für Rückwärtskompatibilität)
      - pk_tabellen_gesamt:         int
      - pk_tabellen_mit_pk:         int
      - fk_referenzen_gesamt:       int
      - fk_referenzen_gueltig:      int
      - fk_referenzen_ungueltig:    int
    """
    syntaktisch_korrekt = _is_syntactically_valid(ddl)

    if not schema.tables:
        return {
            "syntaktisch_korrekt": syntaktisch_korrekt,
            "pk_vollstaendigkeit": 0.0,
            "fk_integritaet": 1.0,
            "pk_tabellen_gesamt": 0,
            "pk_tabellen_mit_pk": 0,
            "fk_referenzen_gesamt": 0,
            "fk_referenzen_gueltig": 0,
            "fk_referenzen_ungueltig": 0,
        }

    n_tabellen = len(schema.tables)
    tables_with_pk = sum(
        1 for t in schema.tables if any(c.primary_key for c in t.columns)
    )
    pk_quote = tables_with_pk / n_tabellen

    table_columns: dict[str, set[str]] = {
        t.name: {c.name for c in t.columns} for t in schema.tables
    }
    total_fks = 0
    valid_fks = 0
    for t in schema.tables:
        for fk in t.foreign_keys:
            total_fks += 1
            if (
                fk.references_table in table_columns
                and fk.references_column in table_columns[fk.references_table]
                and fk.from_column in table_columns[t.name]
            ):
                valid_fks += 1
    fk_quote = 1.0 if total_fks == 0 else valid_fks / total_fks
    invalid_fks = total_fks - valid_fks

    return {
        "syntaktisch_korrekt": syntaktisch_korrekt,
        "pk_vollstaendigkeit": pk_quote,
        "fk_integritaet": fk_quote,
        "pk_tabellen_gesamt": n_tabellen,
        "pk_tabellen_mit_pk": tables_with_pk,
        "fk_referenzen_gesamt": total_fks,
        "fk_referenzen_gueltig": valid_fks,
        "fk_referenzen_ungueltig": invalid_fks,
    }
