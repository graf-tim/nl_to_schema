# NL-to-Schema — Evaluation agentenbasierter Workflow-Architekturen

Begleitendes Implementierungsprojekt zur Bachelorarbeit:

> **Evaluation agentenbasierter Workflow-Architekturen zur automatisierten Generierung relationaler SQL-Schemata aus natürlichsprachlichen Anforderungsbeschreibungen**

---

## Verzeichnisstruktur

```
.
├── testdata_generator/          # Testfalldaten-Generator (GPT-5.4, OpenAI)
│   ├── generator.py             # Erzeugt testcases.json aus NL-Anforderungen
│   ├── validator.py             # Strukturprüfung und Stichprobenauswahl
│   └── output/
│       ├── testcases.json       # 100 generierte Testfälle
│       └── ...
│
└── nl_to_schema/                # Workflow-System und Evaluation
    ├── workflows/               # 5 Workflow-Architekturen (SA, PS, IW, OA, MAD)
    ├── agents/                  # Agenten-Implementierungen (generators, critics, analysts)
    ├── evaluation/              # Strukturelle und semantische Evaluation
    ├── models/                  # Pydantic-Datenmodelle
    ├── run_evaluation.py        # Hauptskript: Workflows ausführen und evaluieren
    ├── run_error_classification.py  # Stichprobenauswahl für manuelle Fehleranalyse
    ├── ddl_generator.py         # Deterministischer DDL-Generator
    └── results/                 # Evaluationsergebnisse
```

---

## Setup

```bash
cd Bachelorarbeit_01
python -m venv .venv
source .venv/bin/activate
pip install -r nl_to_schema/requirements.txt
```

`.env`-Datei im Projektverzeichnis anlegen:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...          # optional, nur für Fehlerklassifikator
WORKFLOW_MODEL=claude-haiku-4-5-20251001
```

---

## Testdaten generieren

```bash
cd testdata_generator

# Alle 100 Testfälle (idempotent — bereits vorhandene werden übersprungen)
python generator.py

# Nur bestimmte Stufen/IDs
python generator.py --limit-per-stufe 5
python generator.py --ids TC001 TC002 TC003
```

---

## Evaluation ausführen

```bash
cd nl_to_schema

# Alle 5 Workflows auf allen Testfällen
python run_evaluation.py \
    --testcases ../testdata_generator/output/testcases.json \
    --output results/

# Nur bestimmte Workflows oder Testfälle
python run_evaluation.py --workflows oa mad --limit-per-stufe 5
python run_evaluation.py --workflows iw --testfall-ids TC001 TC042 --force
```

Ergebnisse werden in `results/<workflow>/<testfall_id>.json` gespeichert,
die Gesamtübersicht in `results/summary.csv`.

**Score-Formel:**

- `score_strukturell = syntax_gate × fk_gate × pk_rate` (Gewicht 40 %)
- `score_semantisch = 0.4·entitaet_f1 + 0.4·attribut_f1 + 0.2·beziehung_recall` (Gewicht 60 %)
- `score_gesamt = 0.40 · score_strukturell + 0.60 · score_semantisch`

---

## Manuelle Fehleranalyse vorbereiten

```bash
cd nl_to_schema

# Stratifizierte Stichprobe (3/3/4 nach Komplexitätsstufe, schlechteste Scores)
python run_error_classification.py \
    --testcases ../testdata_generator/output/testcases.json \
    --results-dir results/
```

Erzeugt in `results/`:

- `manual_review_selection.csv` — 10 ausgewählte Testfälle mit Kontext
- `manual_review.csv` — 50 Zeilen (10 TC × 5 WF) zum manuellen Ausfüllen
- `manual_review.json` — vollständige Daten für die Fehleranalyse
- `executed_manual_review.json` — Durchgeführtes Review

---

## Workflow-Architekturen

| Kürzel | Name                   | Beschreibung                                                            |
| ------ | ---------------------- | ----------------------------------------------------------------------- |
| SA     | Single Agent           | Ein Agent generiert das Schema direkt aus dem Anforderungstext          |
| PS     | Pipeline Sequential    | Dreistufige Pipeline: RA → CMD → LSD (einmalig, kein Feedback)          |
| IW     | Iterativer Workflow    | Generator + generischer Critic in Feedbackschleife (max. 3 Iterationen) |
| OA     | Orchestrator Agent     | Wie PS, aber mit validierungsbasiertem Rücksprung zum Ursprungsschritt  |
| MAD    | Multi-Agent Discussion | Architekt + zwei spezialisierte Critics + Moderator-Synthese            |
