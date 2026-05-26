"""
Summary_Evaluation.py
==================
Reproduziert alle Tabellen und Zahlen aus Kapitel 5 der Bachelorarbeit.

Verwendung:
    python Summary_Evaluation.py --data summary.csv

Abhängigkeiten: Python 3.10+, nur stdlib.
"""

import argparse, csv, math, statistics
from collections import Counter, defaultdict

WORKFLOWS = ['sa', 'ps', 'iw', 'oa', 'mad']
WF       = {'sa':'SA','ps':'PS','iw':'IW','oa':'OA','mad':'MAD'}
STUFEN   = ['1','2','3']
T95      = 1.984   # t-Krit df≈100, zweiseitig 95 %

def load(path):
    return list(csv.DictReader(open(path)))

def m(vals):  return statistics.mean(vals)
def sd(vals): return statistics.stdev(vals)
def ci(vals): return T95 * sd(vals) / math.sqrt(len(vals))

def cohens_d(a, b):
    """d = (mean_b − mean_a) / gepoolte SD.  Positiv → b > a → SA besser."""
    na, nb = len(a), len(b)
    sa_ = sd(a) if na > 1 else 0.0
    sb_ = sd(b) if nb > 1 else 0.0
    p = math.sqrt(((na-1)*sa_**2 + (nb-1)*sb_**2) / (na+nb-2))
    return (m(b) - m(a)) / p if p > 0 else 0.0

def dlabel(d):
    a = abs(d)
    if a < 0.2: return "vernachl."
    if a < 0.5: return "klein"
    if a < 0.8: return "mittel"
    return "gross"

def sep(title):
    print(); print("="*80); print(f"  {title}"); print("="*80)

def hdr(*cols, w=None):
    if w is None: w = [12]*len(cols)
    print("  "+"".join(f"{c:<{x}}" for c,x in zip(cols,w)))
    print("  "+"-"*sum(w))

def prow(*vals, w=None, fmt=".4f"):
    if w is None: w = [12]*len(vals)
    cells = []
    for v,x in zip(vals,w):
        s = f"{v:{fmt}}" if isinstance(v,float) else str(v)
        cells.append(f"{s:<{x}}")
    print("  "+"".join(cells))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    args = ap.parse_args()

    rows = load(args.data)
    print(f"\nGeladen: {len(rows)} Zeilen aus '{args.data}'")

    def sub(wf=None, s=None):
        r = rows
        if wf: r = [x for x in r if x['workflow']==wf]
        if s:  r = [x for x in r if x['stufe']==s]
        return r

    def v(field, wf=None, s=None):
        return [float(x[field]) for x in sub(wf,s)]

    # ── Gesamtscore ─────────────────────────────────────────────
    sep("Gesamtscore je Workflow × Stufe")
    hdr("Workflow","L1","L2","L3","Gesamt", w=[10,10,10,10,10])
    for wf in WORKFLOWS:
        prow(WF[wf],
             m(v('score_gesamt',wf,'1')),
             m(v('score_gesamt',wf,'2')),
             m(v('score_gesamt',wf,'3')),
             m(v('score_gesamt',wf)), w=[10,10,10,10,10])

    # ── Strukturell / Semantisch ────────────────────────────────
    sep("Strukturell / Semantisch (Gesamt)")
    hdr("Workflow","Strukturell","Semantisch","Gesamt", w=[10,13,12,10])
    for wf in WORKFLOWS:
        prow(WF[wf],
             m(v('score_strukturell',wf)),
             m(v('score_semantisch',wf)),
             m(v('score_gesamt',wf)), w=[10,13,12,10])

    # ── Semantische Subdimensionen ──────────────────────────────
    sep("Semantische Subdimensionen (Gesamt)")
    hdr("Workflow","Entität-F1","Attribut-F1","Bez.-Recall", w=[10,13,13,13])
    for wf in WORKFLOWS:
        prow(WF[wf],
             m(v('entitaet_f1',wf)),
             m(v('attribut_f1',wf)),
             m(v('beziehung_recall',wf)), w=[10,13,13,13])

    # ── Subdimensionen SA je Stufe ──────────────────────
    sep("Subdimensionen SA je Stufe (Komplexitätsrückgang)")
    hdr("Stufe","Entität-F1","Attribut-F1","Bez.-Recall","Score-Ges",
        w=[8,13,13,13,11])
    for s in STUFEN:
        prow(f"L{s}",
             m(v('entitaet_f1','sa',s)),
             m(v('attribut_f1','sa',s)),
             m(v('beziehung_recall','sa',s)),
             m(v('score_gesamt','sa',s)), w=[8,13,13,13,11])

    # ── Precision / Recall Entität + Attribut ───────────────────
    sep("Precision und Recall (Entität + Attribut, Gesamt)")
    hdr("Workflow","Ent-P","Ent-R","Attr-P","Attr-R", w=[10,10,10,10,10])
    for wf in WORKFLOWS:
        prow(WF[wf],
             m(v('entitaet_precision',wf)),
             m(v('entitaet_recall',wf)),
             m(v('attribut_precision',wf)),
             m(v('attribut_recall',wf)), w=[10,10,10,10,10])

    # ── Über-/Untergenerierung absolut, Stufe 3 ─────────────────
    sep("Über-/Untergenerierung absolut (Ø pro Schema, Stufe 3)")
    hdr("Workflow","Ent-match","Ent-über","Ent-unter",
        "Attr-über","Attr-unter","Bez-über","Bez-unter",
        w=[10,11,10,11,11,11,9,9])
    for wf in WORKFLOWS:
        prow(WF[wf],
             m(v('entitaet_matched',         wf,'3')),
             m(v('entitaet_nur_in_generiert', wf,'3')),
             m(v('entitaet_nur_in_referenz',  wf,'3')),
             m(v('attribut_nur_in_generiert', wf,'3')),
             m(v('attribut_nur_in_referenz',  wf,'3')),
             m(v('beziehung_nur_in_generiert',wf,'3')),
             m(v('beziehung_nur_in_referenz', wf,'3')),
             w=[10,11,10,11,11,11,9,9], fmt=".2f")

    # Zusatz: Gesamt-Durchschnitt Über-/Untergenerierung alle Stufen
    print("\n  Gesamt (alle Stufen) — Entität:")
    for wf in WORKFLOWS:
        ue = m(v('entitaet_nur_in_generiert',wf))
        un = m(v('entitaet_nur_in_referenz',wf))
        print(f"    {WF[wf]}: über={ue:.2f}  unter={un:.2f}")

    # ── Ressourcenverbrauch ─────────────────────────────────────
    sep("Ressourcenverbrauch (Tokens, LLM-Calls)")
    hdr("Workflow","Tokens Ø","LLM-Calls Ø", w=[10,15,14])
    for wf in WORKFLOWS:
        t = m(v('total_tokens',wf))
        c = m(v('llm_calls',wf))
        print(f"  {WF[wf]:<10}{t:<15,.0f}{c:<14.1f}")
    t_sa  = m(v('total_tokens','sa'))
    t_mad = m(v('total_tokens','mad'))
    print(f"\n  MAD/SA Token-Faktor: {t_mad/t_sa:.1f}×")
    print("\n  Tokens je Stufe:")
    hdr("Workflow","L1","L2","L3", w=[10,13,13,13])
    for wf in WORKFLOWS:
        print(f"  {WF[wf]:<10}", end="")
        for s in STUFEN:
            print(f"{m(v('total_tokens',wf,s)):<13,.0f}", end="")
        print()

    # ── Streuung (SD) ───────────────────────────────────────────
    sep("Streuung des Gesamtscores (Ø ± SD) je Workflow × Stufe")
    hdr("Workflow","L1 Ø±SD","L2 Ø±SD","L3 Ø±SD", w=[10,22,22,22])
    for wf in WORKFLOWS:
        cells = []
        for s in STUFEN:
            vv = v('score_gesamt',wf,s)
            cells.append(f"{m(vv):.3f} ± {sd(vv):.3f}")
        print(f"  {WF[wf]:<10}{cells[0]:<22}{cells[1]:<22}{cells[2]:<22}")

    # ── Cohen's d ───────────────────────────────────────────────
    sep("Cohen's d (SA vs. andere, positiv = SA besser)")
    hdr("Vergleich","L1","L2","L3","Gesamt", w=[14,22,22,22,22])
    sa_all = v('score_gesamt','sa')
    for wf in [x for x in WORKFLOWS if x != 'sa']:
        cells = []
        for s in STUFEN:
            d = cohens_d(v('score_gesamt',wf,s), v('score_gesamt','sa',s))
            cells.append(f"d={d:+.3f} ({dlabel(d)})")
        d_g = cohens_d(v('score_gesamt',wf), sa_all)
        cells.append(f"d={d_g:+.3f} ({dlabel(d_g)})")
        print(f"  SA vs. {WF[wf]:<7}{cells[0]:<22}{cells[1]:<22}"
              f"{cells[2]:<22}{cells[3]:<22}")

    # ── Testfallweises Ranking ──────────────────────────────────
    sep("Testfallweises Ranking (Siege + letzter Platz)")
    tc_scores = defaultdict(dict)
    tc_stufe  = {}
    for r in rows:
        tc_scores[r['testfall_id']][r['workflow']] = float(r['score_gesamt'])
        tc_stufe[r['testfall_id']] = r['stufe']

    wins_tot = Counter(); wins_s = defaultdict(Counter); last = Counter()
    for tc, wd in tc_scores.items():
        s = tc_stufe[tc]
        ranked = sorted(wd.items(), key=lambda x: -x[1])
        wins_tot[ranked[0][0]] += 1
        wins_s[s][ranked[0][0]] += 1
        last[ranked[-1][0]] += 1

    hdr("Workflow","L1","L2","L3","Gesamt","Letzter", w=[10,8,8,8,10,10])
    for wf in WORKFLOWS:
        prow(WF[wf],
             wins_s['1'][wf], wins_s['2'][wf], wins_s['3'][wf],
             wins_tot[wf], last[wf], w=[10,8,8,8,10,10], fmt="d")

    print("\n  SA-Vorsprung je Stufe:")
    hdr("Stufe","n","SA führt","Ø Vorsprung","Min Vorsprung",
        w=[7,5,10,16,16])
    for s in STUFEN:
        tcs = [tc for tc in tc_scores if tc_stufe[tc]==s]
        diffs=[]; leads=0
        for tc in tcs:
            sa_sc = tc_scores[tc].get('sa',0)
            bo = max(vv for k,vv in tc_scores[tc].items() if k!='sa')
            d = sa_sc - bo; diffs.append(d)
            if d > 0: leads += 1
        print(f"  L{s:<6}{len(tcs):<5}{leads}/{len(tcs):<8}"
              f"{m(diffs):+.4f} ({m(diffs)*100:+.2f} pp)  "
              f"{min(diffs):+.4f} ({min(diffs)*100:+.2f} pp)")

    # ── Isolationsvergleiche ───────────────────────────────────
    sep("Isolationsvergleiche: Score-Delta + Token-Faktor")
    pairs = [
        ('sa','ps','SA → PS','F1 (Dekomp.)'),
        ('sa','iw','SA → IW','F2 (iter. Feedback)'),
        ('ps','oa','PS → OA','F2 (Validator)'),
        ('iw','mad','IW → MAD','F2 (deliberativ)'),
    ]
    hdr("Vergleich","Dimension","Score-Delta","Token-Faktor",
        w=[12,20,14,14])
    for a, b, label, dim in pairs:
        delta = (m(v('score_gesamt',b)) - m(v('score_gesamt',a))) * 100
        tf    = m(v('total_tokens',b)) / m(v('total_tokens',a))
        print(f"  {label:<12}{dim:<20}{delta:+.2f} pp{'':<6}×{tf:.1f}")

    # ── Fehlerläufe ──────────────────────────────────────────────────────
    sep("Fehlerbehaftete Läufe")
    err = [r for r in rows if r['error'].strip()]
    print(f"  Gesamt: {len(err)} von {len(rows)}")
    for k,n in Counter(r['error'][:55] for r in err).most_common():
        print(f"    {n}× {k!r}")
    print(f"  Nach Workflow: {dict(Counter(r['workflow'] for r in err))}")
    e_sc = [float(r['score_gesamt']) for r in err]
    o_sc = [float(r['score_gesamt']) for r in rows if not r['error'].strip()]
    print(f"  Ø Score Fehlerläufe:  {m(e_sc):.4f}")
    print(f"  Ø Score fehlerfreie:  {m(o_sc):.4f}")

    print("\n✓ Alle Berechnungen abgeschlossen.")

if __name__ == '__main__':
    main()