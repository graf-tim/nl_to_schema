"""
==================
Verwendung:
    python Summary_Evaluation.py --data summary.csv

Ausgabe:
    Alle Tabellenwerte für Auswertung

Abhängigkeiten: Python 3.10+, keine zusätzlichen Pakete (nur stdlib).
"""

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict

WORKFLOWS = ['sa', 'ps', 'iw', 'oa', 'mad']
WF_LABEL  = {'sa': 'SA', 'ps': 'PS', 'iw': 'IW', 'oa': 'OA', 'mad': 'MAD'}
STUFEN    = ['1', '2', '3']
T_95      = 1.984   # t-Crit für df≈100, zweiseitig 95 %


def load(path: str):
    return list(csv.DictReader(open(path)))


def mean(vals): return statistics.mean(vals)
def sd(vals):   return statistics.stdev(vals)
def ci(vals):   return T_95 * sd(vals) / math.sqrt(len(vals))


def cohens_d(a, b):
    """d = (mean_b - mean_a) / pooled_sd  →  positiv wenn b > a"""
    na, nb = len(a), len(b)
    sa_ = sd(a) if na > 1 else 0.0
    sb_ = sd(b) if nb > 1 else 0.0
    pooled = math.sqrt(((na-1)*sa_**2 + (nb-1)*sb_**2) / (na+nb-2))
    return (mean(b) - mean(a)) / pooled if pooled > 0 else 0.0


def d_label(d):
    a = abs(d)
    if a < 0.2: return "vernachlässigbar"
    if a < 0.5: return "klein"
    if a < 0.8: return "mittel"
    return "gross"


def sep(title=""):
    print()
    print("=" * 80)
    if title: print(f"  {title}")
    print("=" * 80)


def header(*cols, widths=None):
    if widths is None:
        widths = [12] * len(cols)
    print("  " + "".join(f"{c:<{w}}" for c, w in zip(cols, widths)))
    print("  " + "-" * sum(widths))


def row(*vals, widths=None, fmts=None):
    if widths is None:
        widths = [12] * len(vals)
    if fmts is None:
        fmts = [""] * len(vals)
    cells = []
    for v, w, f in zip(vals, widths, fmts):
        if isinstance(v, float):
            s = f"{v:{f}}" if f else f"{v:.4f}"
        else:
            s = str(v)
        cells.append(f"{s:<{w}}")
    print("  " + "".join(cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Pfad zur summary.csv')
    args = parser.parse_args()

    rows = load(args.data)
    print(f"\nGeladen: {len(rows)} Zeilen aus '{args.data}'")

    # Hilfsdatenstrukturen
    def sub(wf=None, s=None):
        r = rows
        if wf: r = [x for x in r if x['workflow'] == wf]
        if s:  r = [x for x in r if x['stufe'] == s]
        return r

    def vals(field, wf=None, s=None):
        return [float(x[field]) for x in sub(wf, s)]

    # ── Gesamtscore ──────────────────────────────────────────
    sep("Mittlerer Gesamtscore je Workflow und Komplexitätsstufe")
    header("Workflow", "L1", "L2", "L3", "Gesamt", widths=[10,10,10,10,10])
    for wf in WORKFLOWS:
        row(WF_LABEL[wf],
            mean(vals('score_gesamt', wf, '1')),
            mean(vals('score_gesamt', wf, '2')),
            mean(vals('score_gesamt', wf, '3')),
            mean(vals('score_gesamt', wf)),
            widths=[10,10,10,10,10])

    # ── Strukturell / Semantisch ────────────────────────────
    sep("Mittlere Scores der strukturellen und semantischen Dimension")
    header("Workflow", "Strukturell", "Semantisch", "Gesamt", widths=[10,13,12,10])
    for wf in WORKFLOWS:
        row(WF_LABEL[wf],
            mean(vals('score_strukturell', wf)),
            mean(vals('score_semantisch', wf)),
            mean(vals('score_gesamt', wf)),
            widths=[10,13,12,10])

    # ── Semantische Subdimensionen ───────────────────────────
    sep("Mittlere Werte der semantischen Subdimensionen")
    header("Workflow", "Entität-F1", "Attribut-F1", "Bez.-Recall", widths=[10,12,13,13])
    for wf in WORKFLOWS:
        row(WF_LABEL[wf],
            mean(vals('entitaet_f1', wf)),
            mean(vals('attribut_f1', wf)),
            mean(vals('beziehung_recall', wf)),
            widths=[10,12,13,13])

    # ── Subdimensionen pro Stufe (SA) ────────────────────
    sep("Subdimensionen SA je Stufe (Komplexitätsabhängigkeit)")
    header("Stufe", "Entität-F1", "Attribut-F1", "Bez.-Recall", "Score-Ges", widths=[8,12,13,13,11])
    for s in STUFEN:
        row(f"L{s}",
            mean(vals('entitaet_f1', 'sa', s)),
            mean(vals('attribut_f1', 'sa', s)),
            mean(vals('beziehung_recall', 'sa', s)),
            mean(vals('score_gesamt', 'sa', s)),
            widths=[8,12,13,13,11])

    # ── Ressourcenverbrauch ──────────────────────────────────
    sep("Mittlerer Token-Verbrauch und LLM-Aufrufe je Workflow")
    header("Workflow", "Tokens Ø", "LLM-Calls Ø", widths=[10,14,14])
    for wf in WORKFLOWS:
        t = mean(vals('total_tokens', wf))
        c = mean(vals('llm_calls', wf))
        print(f"  {WF_LABEL[wf]:<10}{t:<14,.0f}{c:<14.1f}")

    # Token-Faktor MAD/SA
    t_sa  = mean(vals('total_tokens', 'sa'))
    t_mad = mean(vals('total_tokens', 'mad'))
    print(f"\n  MAD/SA Token-Faktor: {t_mad/t_sa:.1f}×")

    # Token je Stufe pro Workflow
    print("\n  Token je Stufe:")
    header("Workflow", "L1", "L2", "L3", widths=[10,12,12,12])
    for wf in WORKFLOWS:
        print(f"  {WF_LABEL[wf]:<10}", end="")
        for s in STUFEN:
            t = mean(vals('total_tokens', wf, s))
            print(f"{t:<12,.0f}", end="")
        print()

    # ── Streuung (SD + 95%-CI) ──────────────────────────────
    sep("Streuung des Gesamtscores (Mittelwert ± SD)")
    header("Workflow", "L1 Ø±SD", "L2 Ø±SD", "L3 Ø±SD", widths=[10,22,22,22])
    for wf in WORKFLOWS:
        row_vals = []
        for s in STUFEN:
            v = vals('score_gesamt', wf, s)
            row_vals.append(f"{mean(v):.3f} ± {sd(v):.3f}")
        print(f"  {WF_LABEL[wf]:<10}{row_vals[0]:<22}{row_vals[1]:<22}{row_vals[2]:<22}")

    # ── Cohen's d gegen SA ──────────────────────────────────
    sep("Cohen's d (SA gegenüber anderen Workflows, positiv = SA besser)")
    header("Vergleich", "L1", "L2", "L3", "Gesamt", widths=[14,22,22,22,22])
    sa_all = vals('score_gesamt', 'sa')
    for wf in [w for w in WORKFLOWS if w != 'sa']:
        cells = []
        for s in STUFEN:
            d = cohens_d(vals('score_gesamt', wf, s), vals('score_gesamt', 'sa', s))
            cells.append(f"d={d:+.3f} ({d_label(d)})")
        d_ges = cohens_d(vals('score_gesamt', wf), sa_all)
        cells.append(f"d={d_ges:+.3f} ({d_label(d_ges)})")
        print(f"  SA vs. {WF_LABEL[wf]:<7}{cells[0]:<22}{cells[1]:<22}{cells[2]:<22}{cells[3]:<22}")

    # ── Testfallweises Ranking ──────────────────────────────
    sep("Anzahl Testfälle mit höchstem Gesamtscore je Workflow und Stufe")
    # Aufbau: tc -> {wf: score}
    tc_scores = defaultdict(dict)
    tc_stufe  = {}
    for r in rows:
        tc_scores[r['testfall_id']][r['workflow']] = float(r['score_gesamt'])
        tc_stufe[r['testfall_id']] = r['stufe']

    wins_total = Counter()
    wins_stufe = defaultdict(Counter)
    last_total = Counter()

    for tc, wf_d in tc_scores.items():
        s = tc_stufe[tc]
        ranked = sorted(wf_d.items(), key=lambda x: -x[1])
        wins_total[ranked[0][0]] += 1
        wins_stufe[s][ranked[0][0]] += 1
        last_total[ranked[-1][0]] += 1

    header("Workflow", "L1", "L2", "L3", "Gesamt", "Letzter", widths=[10,8,8,8,10,10])
    for wf in WORKFLOWS:
        row(WF_LABEL[wf],
            wins_stufe['1'][wf],
            wins_stufe['2'][wf],
            wins_stufe['3'][wf],
            wins_total[wf],
            last_total[wf],
            widths=[10,8,8,8,10,10])

    # SA-Vorsprung je Testfall
    print("\n  SA-Vorsprung je Stufe (Score SA minus bester anderer WF):")
    header("Stufe", "n Testfälle", "SA führt", "Ø Vorsprung", "Min Vorsprung", widths=[8,13,10,15,15])
    for s in STUFEN:
        tc_s = [tc for tc in tc_scores if tc_stufe[tc] == s]
        diffs = []
        leads = 0
        for tc in tc_s:
            sa_sc = tc_scores[tc].get('sa', 0)
            best_other = max(v for k, v in tc_scores[tc].items() if k != 'sa')
            diff = sa_sc - best_other
            diffs.append(diff)
            if diff > 0: leads += 1
        print(f"  L{s:<7}{len(tc_s):<13}{leads}/{len(tc_s):<8}{mean(diffs):+.4f} ({mean(diffs)*100:+.2f} pp){min(diffs):+.4f} ({min(diffs)*100:+.2f} pp)")

    # ── Isolationsvergleiche ────────────────────────────────
    sep("Isolationsvergleiche: Score-Delta und Token-Faktor")
    pairs = [
        ('sa','ps','SA → PS','F1 (Dekomp.)'),
        ('sa','iw','SA → IW','F2 (iter. Feedback)'),
        ('ps','oa','PS → OA','F2 (Validator)'),
        ('iw','mad','IW → MAD','F2 (deliberativ)'),
    ]
    header("Vergleich", "Dimension", "Score-Delta", "Token-Faktor", widths=[12,18,14,14])
    for a, b, label, dim in pairs:
        delta = (mean(vals('score_gesamt', b)) - mean(vals('score_gesamt', a))) * 100
        tf    = mean(vals('total_tokens', b)) / mean(vals('total_tokens', a))
        print(f"  {label:<12}{dim:<18}{delta:+.2f} pp{' ':6}×{tf:.1f}")

    # ── FEHLERLÄUFE ──────────────────────────────────────────────────────
    sep("Fehlerbehaftete Läufe")
    err = [r for r in rows if r['error'].strip()]
    print(f"  Gesamt fehlermarkiert: {len(err)} von {len(rows)}")
    err_types = Counter(r['error'][:50] for r in err)
    for k, v in err_types.most_common():
        print(f"    {v}× {k!r}")
    err_wf = Counter(r['workflow'] for r in err)
    print(f"  Nach Workflow: {dict(err_wf)}")
    err_scores = [float(r['score_gesamt']) for r in err]
    ok_scores  = [float(r['score_gesamt']) for r in rows if not r['error'].strip()]
    print(f"  Ø Score Fehlerläufe: {mean(err_scores):.4f}")
    print(f"  Ø Score fehlerfreie: {mean(ok_scores):.4f}")

    print("\n✓ Alle Berechnungen abgeschlossen.")


if __name__ == '__main__':
    main()