"""Semantische Evaluation per Embedding-Matching."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from models.schema import LogicalSchema, Table


logger = logging.getLogger(__name__)


# Bereich für unsichere Matches (manuelle Nachkontrolle).
GREY_ZONE_LOW = 0.60
GREY_ZONE_HIGH = 0.75


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
    sim, names_gen: list[str], names_ref: list[str], threshold: float, grey_log: list
):
    """Greedy-Matching: pro Zeile bestes verbliebenes Ziel ab Threshold."""
    pairs: list[tuple[int, int, float]] = []
    used_ref: set[int] = set()
    n_gen = len(names_gen)
    n_ref = len(names_ref)
    if n_gen == 0 or n_ref == 0:
        return pairs

    flat = []
    for i in range(n_gen):
        for j in range(n_ref):
            flat.append((float(sim[i][j]), i, j))
    flat.sort(reverse=True)
    used_gen: set[int] = set()
    for score, i, j in flat:
        if i in used_gen or j in used_ref:
            continue
        if score < GREY_ZONE_LOW:
            continue
        if GREY_ZONE_LOW <= score < GREY_ZONE_HIGH:
            grey_log.append(
                {
                    "generated": names_gen[i],
                    "reference": names_ref[j],
                    "similarity": score,
                }
            )
            continue
        if score >= threshold:
            pairs.append((i, j, score))
            used_gen.add(i)
            used_ref.add(j)
    return pairs


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _attribute_f1_for_match(
    gen: Table, ref: Table, threshold: float, grey_log: list
) -> tuple[int, int, int]:
    gen_cols = [c.name for c in gen.columns]
    ref_cols = [c.name for c in ref.columns]
    if not gen_cols or not ref_cols:
        tp = 0
        fp = len(gen_cols)
        fn = len(ref_cols)
        return tp, fp, fn

    sim = _cosine_matrix(_embed(gen_cols), _embed(ref_cols))
    pairs = _greedy_match(sim, gen_cols, ref_cols, threshold, grey_log)
    tp = len(pairs)
    fp = len(gen_cols) - tp
    fn = len(ref_cols) - tp
    return tp, fp, fn


def _relationships(schema: LogicalSchema) -> set[tuple[str, str]]:
    rels: set[tuple[str, str]] = set()
    for t in schema.tables:
        for fk in t.foreign_keys:
            rels.add((t.name.lower(), fk.to_table.lower()))
    return rels


def semantic_score(
    generated: LogicalSchema,
    reference: LogicalSchema,
    threshold: float = 0.75,
) -> dict:
    """Embedding-basiertes Matching gegen ein Referenzschema.

    Rückgabe-Dict:
      entitaet_f1:       F1 auf Tabellennamen
      attribut_f1:       gewichtetes Mittel über gematchte Tabellen-Paare
      beziehung_recall:  Recall auf FK-Beziehungen (basierend auf gematchten Tabellen)
      semantic_score:    0.4 * entitaet_f1 + 0.4 * attribut_f1 + 0.2 * beziehung_recall
      grey_zone:         Liste unsicherer Matches (manuelle Kontrolle)
    """
    grey_log: list[dict] = []

    gen_names = [t.name for t in generated.tables]
    ref_names = [t.name for t in reference.tables]
    sim = _cosine_matrix(_embed(gen_names), _embed(ref_names))
    table_pairs = _greedy_match(sim, gen_names, ref_names, threshold, grey_log)

    tp = len(table_pairs)
    fp = len(gen_names) - tp
    fn = len(ref_names) - tp
    entitaet_f1 = _f1(tp, fp, fn)

    attr_tp = attr_fp = attr_fn = 0
    name_to_table_gen = {t.name: t for t in generated.tables}
    name_to_table_ref = {t.name: t for t in reference.tables}
    matched_pairs_by_name: dict[str, str] = {}
    for i, j, _ in table_pairs:
        gen_t = name_to_table_gen[gen_names[i]]
        ref_t = name_to_table_ref[ref_names[j]]
        matched_pairs_by_name[gen_t.name.lower()] = ref_t.name.lower()
        a_tp, a_fp, a_fn = _attribute_f1_for_match(gen_t, ref_t, threshold, grey_log)
        attr_tp += a_tp
        attr_fp += a_fp
        attr_fn += a_fn
    attribut_f1 = _f1(attr_tp, attr_fp, attr_fn)

    ref_rels = _relationships(reference)
    gen_rels = _relationships(generated)
    if not ref_rels:
        beziehung_recall = 1.0
    else:
        # Mappe gen-Relationen via gematchter Tabellennamen ins Referenzraum.
        translated: set[tuple[str, str]] = set()
        for src, tgt in gen_rels:
            tsrc = matched_pairs_by_name.get(src)
            ttgt = matched_pairs_by_name.get(tgt)
            if tsrc and ttgt:
                translated.add((tsrc, ttgt))
        hit = sum(1 for r in ref_rels if r in translated)
        beziehung_recall = hit / len(ref_rels)

    score = 0.4 * entitaet_f1 + 0.4 * attribut_f1 + 0.2 * beziehung_recall

    return {
        "entitaet_f1": entitaet_f1,
        "attribut_f1": attribut_f1,
        "beziehung_recall": beziehung_recall,
        "semantic_score": score,
        "grey_zone": grey_log,
    }
