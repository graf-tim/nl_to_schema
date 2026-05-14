"""Semantische Evaluation per Embedding-Matching.

Bewertungslogik (unverändert):
  - all-MiniLM-L6-v2 Embeddings
  - Greedy-Matching mit hartem Threshold 0.75 (keine Grauzone)
  - Tabellen-, Attribut- und Beziehungs-Matching wie bisher
  - Score = 0.4·entitaet_f1 + 0.4·attribut_f1 + 0.2·beziehung_recall

Output-Erweiterung:
  - Pro Dimension: precision/recall/f1 plus absolute Zählwerte
    (matched, nur_in_referenz, nur_in_generiert)
  - Abweichungs-Liste: alle nicht-gematchten Referenzelemente, jeweils mit
    fehlerklasse=None (wird durch error_classifier befüllt)
"""
from __future__ import annotations

import logging
from functools import lru_cache

from models.schema import LogicalSchema, Table


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def _embed(texts: list[str]):
    import numpy as np

    if not texts:
        return np.zeros((0, 384), dtype=float)
    model = _model()
    return model.encode(texts, normalize_embeddings=True)


def _cosine_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        import numpy as np

        return np.zeros((len(a), len(b)), dtype=float)
    return a @ b.T


def _greedy_match(
    sim,
    names_gen: list[str],
    names_ref: list[str],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedy-Matching mit hartem Threshold (keine Grauzone)."""
    n_gen = len(names_gen)
    n_ref = len(names_ref)
    if n_gen == 0 or n_ref == 0:
        return []

    flat: list[tuple[float, int, int]] = []
    for i in range(n_gen):
        for j in range(n_ref):
            s = float(sim[i][j])
            if s >= threshold:
                flat.append((s, i, j))
    flat.sort(reverse=True)

    pairs: list[tuple[int, int, float]] = []
    used_gen: set[int] = set()
    used_ref: set[int] = set()
    for score, i, j in flat:
        if i in used_gen or j in used_ref:
            continue
        pairs.append((i, j, score))
        used_gen.add(i)
        used_ref.add(j)
    return pairs


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    if tp == 0:
        return 0.0, 0.0, 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _attribute_pairs(
    gen: Table, ref: Table, threshold: float
) -> tuple[list[tuple[int, int, float]], list[str], list[str]]:
    """Greedy-Match Spaltennamen innerhalb eines gematchten Tabellenpaars.

    Rückgabe: (pairs, gen_cols, ref_cols).
    """
    gen_cols = [c.name for c in gen.columns]
    ref_cols = [c.name for c in ref.columns]
    if not gen_cols or not ref_cols:
        return [], gen_cols, ref_cols
    sim = _cosine_matrix(_embed(gen_cols), _embed(ref_cols))
    pairs = _greedy_match(sim, gen_cols, ref_cols, threshold)
    return pairs, gen_cols, ref_cols


def _relationships(schema: LogicalSchema) -> set[tuple[str, str]]:
    rels: set[tuple[str, str]] = set()
    for t in schema.tables:
        for fk in t.foreign_keys:
            rels.add((t.name.lower(), fk.references_table.lower()))
    return rels


def semantic_score(
    generated: LogicalSchema,
    reference: LogicalSchema,
    threshold: float = 0.75,
) -> dict:
    """Embedding-basiertes Matching gegen ein Referenzschema.

    Rückgabe-Dict siehe Modul-Docstring.
    """
    # ---- Tabellen ----
    gen_names = [t.name for t in generated.tables]
    ref_names = [t.name for t in reference.tables]
    sim = _cosine_matrix(_embed(gen_names), _embed(ref_names))
    table_pairs = _greedy_match(sim, gen_names, ref_names, threshold)

    matched_gen_ix = {i for i, _, _ in table_pairs}
    matched_ref_ix = {j for _, j, _ in table_pairs}
    matched_tabellen = len(table_pairs)
    nur_in_ref_tabellen = len(ref_names) - matched_tabellen
    nur_in_gen_tabellen = len(gen_names) - matched_tabellen
    e_prec, e_rec, e_f1 = _precision_recall_f1(
        tp=matched_tabellen, fp=nur_in_gen_tabellen, fn=nur_in_ref_tabellen
    )

    # ---- Attribute ----
    by_name_gen = {t.name: t for t in generated.tables}
    by_name_ref = {t.name: t for t in reference.tables}
    matched_pairs_by_name: dict[str, str] = {}
    for i, j, _ in table_pairs:
        matched_pairs_by_name[gen_names[i].lower()] = ref_names[j].lower()

    attr_tp = 0
    attr_fp = 0
    attr_fn = 0
    # Nicht-gematchte Referenz-Attribute (für Abweichungen):
    nicht_gematchte_attribute: list[str] = []  # Format: "table.column"

    # Attribute aus gematchten Tabellen-Paaren
    matched_ref_tables_for_attrs: set[str] = set()
    for i, j, _ in table_pairs:
        gen_t = by_name_gen[gen_names[i]]
        ref_t = by_name_ref[ref_names[j]]
        matched_ref_tables_for_attrs.add(ref_t.name)
        pairs, gen_cols, ref_cols = _attribute_pairs(gen_t, ref_t, threshold)
        a_tp = len(pairs)
        a_fp = len(gen_cols) - a_tp
        a_fn = len(ref_cols) - a_tp
        attr_tp += a_tp
        attr_fp += a_fp
        attr_fn += a_fn

        matched_ref_col_ix = {jj for _, jj, _ in pairs}
        for jj, col_name in enumerate(ref_cols):
            if jj not in matched_ref_col_ix:
                nicht_gematchte_attribute.append(f"{ref_t.name}.{col_name}")

    # Attribute aus nicht-gematchten Referenz-Tabellen: granular alle aufführen.
    for j, ref_t in enumerate(reference.tables):
        if j in matched_ref_ix:
            continue
        attr_fn += len(ref_t.columns)
        for col in ref_t.columns:
            nicht_gematchte_attribute.append(f"{ref_t.name}.{col.name}")

    # Spalten aus nicht-gematchten generierten Tabellen → reine Generated-Spalten
    for i, gen_t in enumerate(generated.tables):
        if i in matched_gen_ix:
            continue
        attr_fp += len(gen_t.columns)

    attr_matched = attr_tp
    attr_nur_ref = attr_fn
    attr_nur_gen = attr_fp
    a_prec, a_rec, a_f1 = _precision_recall_f1(
        tp=attr_matched, fp=attr_nur_gen, fn=attr_nur_ref
    )

    # ---- Beziehungen ----
    ref_rels = _relationships(reference)
    gen_rels = _relationships(generated)

    # Generated-Relationen über Tabellen-Matching ins Referenzraum übersetzen
    translated_gen_rels: set[tuple[str, str]] = set()
    for src, tgt in gen_rels:
        tsrc = matched_pairs_by_name.get(src)
        ttgt = matched_pairs_by_name.get(tgt)
        if tsrc and ttgt:
            translated_gen_rels.add((tsrc, ttgt))

    matched_rels = ref_rels & translated_gen_rels
    nur_in_ref_rels = ref_rels - translated_gen_rels
    nur_in_gen_rels = translated_gen_rels - ref_rels

    if not ref_rels:
        beziehung_recall = 1.0
    else:
        beziehung_recall = len(matched_rels) / len(ref_rels)

    # ---- Score ----
    score = 0.4 * e_f1 + 0.4 * a_f1 + 0.2 * beziehung_recall

    # ---- Abweichungen (alle nicht-gematchten Referenzelemente) ----
    abweichungen: list[dict] = []
    for j, ref_t in enumerate(reference.tables):
        if j not in matched_ref_ix:
            abweichungen.append(
                {"element": ref_t.name, "typ": "tabelle", "fehlerklasse": None}
            )
    for elem in nicht_gematchte_attribute:
        abweichungen.append(
            {"element": elem, "typ": "attribut", "fehlerklasse": None}
        )
    for src, tgt in sorted(nur_in_ref_rels):
        abweichungen.append(
            {"element": f"{src} → {tgt}", "typ": "beziehung", "fehlerklasse": None}
        )

    return {
        "score": score,
        "entitaet": {
            "precision": e_prec,
            "recall": e_rec,
            "f1": e_f1,
            "matched": matched_tabellen,
            "nur_in_referenz": nur_in_ref_tabellen,
            "nur_in_generiert": nur_in_gen_tabellen,
        },
        "attribut": {
            "precision": a_prec,
            "recall": a_rec,
            "f1": a_f1,
            "matched": attr_matched,
            "nur_in_referenz": attr_nur_ref,
            "nur_in_generiert": attr_nur_gen,
        },
        "beziehung": {
            "recall": beziehung_recall,
            "matched": len(matched_rels),
            "nur_in_referenz": len(nur_in_ref_rels),
            "nur_in_generiert": len(nur_in_gen_rels),
        },
        "abweichungen": abweichungen,
    }
