# Privplay

PHI/PII Classification Engine with Training Pipeline.

## Detection Stack (Defense-in-Depth)

Privplay uses multiple detection layers for maximum HIPAA coverage:

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Transformer │   │  Presidio   │   │    Rules    │   │    LLM      │
│ (Stanford)  │ → │  + Dicts    │ → │   (Regex)   │ → │  Verifier   │
│ PHI-specific│   │  PII + Med  │   │ HIPAA 18 IDs│   │  Uncertain  │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
```

### HIPAA 18 Identifier Coverage

| # | Identifier | Detection Method |
|---|------------|------------------|
| 1 | Names | Transformer + Presidio NER |
| 2 | Geographic | Presidio + Rules |
| 3 | Dates | Transformer + Presidio + Rules |
| 4 | Phone | Presidio + Rules |
| 5 | Fax | Rules (labeled) |
| 6 | Email | Presidio + Rules |
| 7 | SSN | Presidio + Rules |
| 8 | MRN | Transformer + Rules |
| 9 | Health Plan ID | Rules (MBI, Member ID patterns) |
| 10 | Account Numbers | Presidio + Rules |
| 11 | License/DEA | Rules (DEA, Medical License) |
| 12 | Vehicle ID (VIN) | Rules |
| 13 | Device ID (UDI) | Rules |
| 14 | URLs | Presidio + Rules |
| 15 | IP Addresses | Presidio + Rules |
| 16-18 | Biometric/Photos/Other | N/A (non-text) |

### Dictionary-Based Detection

| Dictionary | Entity Type | Source |
|------------|-------------|--------|
| Drug Names | DRUG | FDA NDC (~30K) |
| Hospital Names | HOSPITAL | CMS Provider (~6K) |
| Insurance Payers | HEALTH_PLAN | Bundled (~100) |
| Lab Tests | LAB_TEST | Bundled (~300) |
| Provider NPIs | NPI | NPPES (~8M, SQLite) |

## Quick Start

```bash
# Install
cd privplay
pip install -e ".[presidio]"
python -m spacy download en_core_web_sm

# Download dictionaries (required)
phi-train download all

# Generate synthetic training data
phi-train faker -n 50

# Check detection stack status
phi-train stack

# Scan for PHI/PII
phi-train scan

# Interactive review session
phi-train review

# Check progress
phi-train stats

# Run F1 evaluation
phi-train test
```

## Commands

| Command | Description |
|---------|-------------|
| `phi-train faker` | Generate synthetic clinical documents |
| `phi-train import <path>` | Import documents from files |
| `phi-train scan` | Scan documents for PHI/PII |
| `phi-train review` | Interactive review session |
| `phi-train stats` | Show training progress |
| `phi-train test` | Run F1 evaluation |
| `phi-train detect <text>` | Detect PHI in single text |
| `phi-train download <target>` | Download dictionaries/NPI |
| `phi-train stack` | Show detection stack status |
| `phi-train export <file>` | Export corrections |
| `phi-train reset` | Reset database |
| `phi-train config-show` | Show configuration |

## Workflow

1. **Install**: `pip install -e ".[presidio]"` + spacy model
2. **Download**: `phi-train download all` (dictionaries + NPI)
3. **Generate**: `phi-train faker -n 100`
4. **Scan**: `phi-train scan` (full detection stack)
5. **Review**: `phi-train review` (approve/reject below 95%)
6. **Evaluate**: `phi-train test` (see F1 score)
7. **Iterate**: Keep reviewing to improve accuracy

## Configuration

Config file: `~/.privplay/config.yaml`

```yaml
confidence_threshold: 0.95
context_window: 50

presidio:
  enabled: true
  score_threshold: 0.5

verification:
  provider: ollama
  ollama:
    url: http://localhost:11434
    model: phi3:mini
```

## Entity Types

**Identity**: NAME_PERSON, NAME_PATIENT, NAME_PROVIDER, SSN, MRN, NPI, DEA_NUMBER, MEDICAL_LICENSE
**Contact**: EMAIL, PHONE, FAX, ADDRESS, ZIP
**Temporal**: DATE, DATE_DOB, AGE
**Financial**: CREDIT_CARD, BANK_ACCOUNT
**Digital**: IP_ADDRESS, URL, USERNAME, VIN, UDI
**Clinical**: DRUG, LAB_TEST, HOSPITAL, HEALTH_PLAN, HEALTH_PLAN_ID
