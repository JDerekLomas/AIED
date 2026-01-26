# Eedi Dataset Access and Structure Guide

## Dataset Source

**Kaggle Competition:** [Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)

**Access Requirements:**
- Kaggle account
- Accept competition rules (even though competition ended)
- Download via Kaggle API or web interface

---

## How to Download

### Option 1: Kaggle Web Interface

1. Go to https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/data
2. Sign in to Kaggle
3. Accept competition rules
4. Click "Download All" or download individual files

### Option 2: Kaggle API

```bash
# Install Kaggle CLI if needed
pip install kaggle

# Ensure API key is configured (~/.kaggle/kaggle.json)

# Download competition data
kaggle competitions download -c eedi-mining-misconceptions-in-mathematics

# Unzip
unzip eedi-mining-misconceptions-in-mathematics.zip -d data/eedi/
```

---

## Expected File Structure

Based on competition documentation and winner solutions:

```
data/eedi/
├── train.csv                    # Training items with misconception labels
├── test.csv                     # Test items (no labels)
├── misconception_mapping.csv    # Maps MisconceptionId to MisconceptionName
├── sample_submission.csv        # Submission format
└── (possibly additional files)
```

---

## Key Data Schema

### train.csv

| Column | Type | Description |
|--------|------|-------------|
| QuestionId | int | Unique question identifier |
| ConstructId | int | ID of the math construct being tested |
| ConstructName | str | Name of construct (e.g., "Order of Operations") |
| SubjectId | int | Subject area ID |
| SubjectName | str | Subject area name |
| QuestionText | str | The question stem |
| CorrectAnswer | str | Which option is correct (A, B, C, or D) |
| AnswerAText | str | Text of option A |
| AnswerBText | str | Text of option B |
| AnswerCText | str | Text of option C |
| AnswerDText | str | Text of option D |
| MisconceptionAId | int/null | Misconception ID for distractor A (null if correct) |
| MisconceptionBId | int/null | Misconception ID for distractor B (null if correct) |
| MisconceptionCId | int/null | Misconception ID for distractor C (null if correct) |
| MisconceptionDId | int/null | Misconception ID for distractor D (null if correct) |

### misconception_mapping.csv

| Column | Type | Description |
|--------|------|-------------|
| MisconceptionId | int | Unique misconception identifier |
| MisconceptionName | str | Description of the misconception |

---

## Data Characteristics

Based on competition description:

| Attribute | Value |
|-----------|-------|
| Total questions | ~1,868 |
| Format | Multiple choice (4 options) |
| Domain | Mathematics (grades 4-8) |
| Languages | English |
| Misconception taxonomy | ~2,000+ unique misconceptions |
| Labeling | Expert-annotated by 15 trained math tutors |

---

## Sample Data Structure

### Example Item

```json
{
  "QuestionId": 12345,
  "ConstructId": 42,
  "ConstructName": "Order of Operations with Brackets",
  "SubjectId": 1,
  "SubjectName": "Number",
  "QuestionText": "Calculate 3 + 2 × 4",
  "CorrectAnswer": "B",
  "AnswerAText": "20",
  "AnswerBText": "11",
  "AnswerCText": "9",
  "AnswerDText": "14",
  "MisconceptionAId": 1234,
  "MisconceptionBId": null,
  "MisconceptionCId": 5678,
  "MisconceptionDId": 9012
}
```

### Example Misconception

```json
{
  "MisconceptionId": 1234,
  "MisconceptionName": "Carries out operations from left to right regardless of priority order"
}
```

---

## Data Processing for Study 2

### Step 1: Load and Merge

```python
import pandas as pd

# Load data
train = pd.read_csv('data/eedi/train.csv')
misconceptions = pd.read_csv('data/eedi/misconception_mapping.csv')

# Create lookup dict
misconception_lookup = dict(zip(
    misconceptions['MisconceptionId'],
    misconceptions['MisconceptionName']
))
```

### Step 2: Reshape to Long Format

```python
def reshape_item(row):
    """Convert wide format to long format with one row per distractor."""
    items = []
    for option in ['A', 'B', 'C', 'D']:
        misconception_id = row[f'Misconception{option}Id']
        if pd.notna(misconception_id):  # Skip correct answer
            items.append({
                'QuestionId': row['QuestionId'],
                'ConstructName': row['ConstructName'],
                'QuestionText': row['QuestionText'],
                'CorrectAnswer': row['CorrectAnswer'],
                'DistractorOption': option,
                'DistractorText': row[f'Answer{option}Text'],
                'MisconceptionId': int(misconception_id),
                'MisconceptionName': misconception_lookup.get(int(misconception_id), 'Unknown'),
                'AnswerA': row['AnswerAText'],
                'AnswerB': row['AnswerBText'],
                'AnswerC': row['AnswerCText'],
                'AnswerD': row['AnswerDText'],
            })
    return items

# Apply to all rows
long_format = []
for _, row in train.iterrows():
    long_format.extend(reshape_item(row))

df_long = pd.DataFrame(long_format)
```

### Step 3: Add Misconception Categories

```python
# Top-level category mapping (manual or from documentation)
PROCEDURAL_KEYWORDS = ['carries out', 'forgets to', 'adds instead', 'subtracts instead',
                       'multiplies', 'divides', 'wrong operation', 'calculation']
CONCEPTUAL_KEYWORDS = ['believes', 'thinks that', 'confuses', 'does not understand',
                       'misconception about']
INTERPRETIVE_KEYWORDS = ['misreads', 'reads', 'identifies wrong', 'misinterprets']

def categorize_misconception(name):
    name_lower = name.lower()
    if any(kw in name_lower for kw in PROCEDURAL_KEYWORDS):
        return 'Procedural'
    elif any(kw in name_lower for kw in CONCEPTUAL_KEYWORDS):
        return 'Conceptual'
    elif any(kw in name_lower for kw in INTERPRETIVE_KEYWORDS):
        return 'Interpretive'
    else:
        return 'Other'

df_long['MisconceptionCategory'] = df_long['MisconceptionName'].apply(categorize_misconception)
```

### Step 4: Sample Selection

```python
def select_study_sample(df, n_per_category=133):
    """Select balanced sample across misconception categories."""
    sample = df.groupby('MisconceptionCategory').apply(
        lambda x: x.sample(n=min(len(x), n_per_category), random_state=42)
    ).reset_index(drop=True)
    return sample

# Target: 400 items (133 per category × 3 categories + buffer)
study_sample = select_study_sample(df_long)
```

---

## Key Fields for Study 2

| Field | Use in Study |
|-------|--------------|
| QuestionId | Item identifier |
| QuestionText | Input to LLM |
| Answer[A-D]Text | Options for LLM |
| CorrectAnswer | Ground truth |
| MisconceptionId | Link to misconception |
| MisconceptionName | Ground truth for coding |
| MisconceptionCategory | Stratification variable |

---

## Data Quality Checks

Before running study:

```python
# Check for missing values
print(train.isnull().sum())

# Check misconception coverage
print(f"Unique misconceptions: {df_long['MisconceptionId'].nunique()}")
print(f"Items with all distractors labeled: {(train['MisconceptionAId'].notna() | ...)}...")

# Check category balance
print(df_long['MisconceptionCategory'].value_counts())

# Sample items for manual review
print(df_long.sample(5)[['QuestionText', 'DistractorText', 'MisconceptionName']])
```

---

## Alternative: If Eedi Access Fails

### ASSISTments Backup

```bash
# Download ASSISTments 2012
wget https://sites.google.com/site/assistmentsdata/...
```

Process similarly, but will need to:
1. Compute distractor frequencies from raw attempt data
2. Manually code misconception types (no pre-existing labels)

### Generated Items Backup

Use probe items (see probe_items.md) as primary dataset if external data unavailable.

---

## References

- [Kaggle Competition](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)
- [Eedi Blog Post](https://www.eedi.com/news/from-wrong-answers-to-real-insights-how-we-used-a-kaggle-challenge-to-map-student-misconceptions)
- [NeurIPS 2020 Eedi Challenge](https://eedi.com/projects/neurips-education-challenge)

---

*Version 1.0 - January 26, 2026*
