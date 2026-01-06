# Raw Datasets

## LongBench

| Dataset | File | Description |
|---------|------|-------------|
| 2WikiMQA | `2wikimqa.jsonl` | Multi-hop QA from Wikipedia |
| HotpotQA | `hotpotqa.jsonl` | Multi-hop QA requiring reasoning |
| MuSiQue | `musique.jsonl` | Multi-hop QA with decomposed questions |
| NarrativeQA | `narrativeqa.jsonl` | QA on narrative texts |
| Qasper | `qasper.jsonl` | QA on scientific papers |

## UltraDomain

| Dataset | File | Description |
|---------|------|-------------|
| Biology | `biology.jsonl` | Domain-specific QA |
| Fiction | `fiction.jsonl` | Domain-specific QA |
| Music | `music.jsonl` | Domain-specific QA |
| Philosophy | `philosophy.jsonl` | Domain-specific QA |
| Technology | `technology.jsonl` | Domain-specific QA |

## Training

| Dataset | File | Description |
|---------|------|-------------|
| Finance | `fin.jsonl` | Financial domain QA |
| Legal | `legal.jsonl` | Legal domain QA |

## Format

Each JSONL file contains records with:
- `input`: Question
- `answers`: List of answer strings
- `context`: Document context
