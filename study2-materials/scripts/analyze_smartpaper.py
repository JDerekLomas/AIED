"""
Classical test theory + IRT analysis of SmartPaper item response data.
Prepares ground-truth difficulty/discrimination for LLM estimation experiments.
"""
import csv
import numpy as np
from collections import defaultdict
import json

# Load responses
responses = []
with open('/Users/dereklomas/AIED/study2-materials/data/smartpaper/export_item_responses.csv') as f:
    responses = list(csv.DictReader(f))

# Load metadata
metadata = {}
with open('/Users/dereklomas/AIED/study2-materials/data/smartpaper/export_assessment_metadata.csv') as f:
    for row in csv.DictReader(f):
        key = (row['AssessmentName'], row['QuestionNumber'])
        metadata[key] = row

print(f"Loaded {len(responses)} responses, {len(metadata)} items\n")

# Build student x item matrix per assessment
# Unique student = SubOrganizationName + GroupId + AssessmentName + RollNumber
assessments = defaultdict(lambda: defaultdict(dict))
for r in responses:
    student_id = (r['SubOrganizationName'], r['GroupId'], r['RollNumber'])
    assessment = r['AssessmentName']
    q = int(r['QuestionNumber'])
    score = float(r['StudentScore']) if r['StudentScore'] else 0
    max_score = float(r['MaxScore']) if r['MaxScore'] else 1
    # Store proportion of max score (0-1)
    assessments[assessment][(student_id, q)] = score / max_score if max_score > 0 else 0

# For each assessment, build item statistics
all_items = []

for assessment_name in sorted(assessments.keys()):
    # Get all students and questions
    students = set()
    questions = set()
    for (sid, q) in assessments[assessment_name]:
        students.add(sid)
        questions.add(q)
    
    students = sorted(students)
    questions = sorted(questions)
    
    # Build matrix (students x items), NaN for missing
    n_students = len(students)
    n_items = len(questions)
    student_idx = {s: i for i, s in enumerate(students)}
    item_idx = {q: i for i, q in enumerate(questions)}
    
    matrix = np.full((n_students, n_items), np.nan)
    for (sid, q), score in assessments[assessment_name].items():
        if sid in student_idx and q in item_idx:
            matrix[student_idx[sid], item_idx[q]] = score
    
    # Classical item statistics
    print(f"=== {assessment_name} ===")
    print(f"  Students: {n_students}, Items: {n_items}")
    print(f"  Response rate: {np.sum(~np.isnan(matrix))/(n_students*n_items):.1%}")
    
    # Total scores per student (mean of non-missing)
    student_totals = np.nanmean(matrix, axis=1)
    
    # Cronbach's alpha
    valid_mask = ~np.isnan(matrix)
    # Use complete cases only for alpha
    complete = ~np.any(np.isnan(matrix), axis=1)
    if np.sum(complete) > 10:
        mat_complete = matrix[complete]
        k = mat_complete.shape[1]
        item_vars = np.var(mat_complete, axis=0, ddof=1)
        total_var = np.var(np.sum(mat_complete, axis=1), ddof=1)
        alpha = (k / (k - 1)) * (1 - np.sum(item_vars) / total_var) if total_var > 0 else 0
        print(f"  Cronbach's alpha: {alpha:.3f} (n={np.sum(complete)} complete cases)")
    
    for q in questions:
        j = item_idx[q]
        item_scores = matrix[:, j]
        valid = item_scores[~np.isnan(item_scores)]
        
        if len(valid) < 10:
            continue
        
        # Classical difficulty (mean score, 0-1 scale)
        p = np.mean(valid)
        
        # Point-biserial correlation (item-total correlation)
        # Use students who answered this item
        valid_students = ~np.isnan(item_scores)
        totals_for_valid = student_totals[valid_students]
        scores_for_valid = item_scores[valid_students]
        
        if np.std(scores_for_valid) > 0 and np.std(totals_for_valid) > 0:
            rpb = np.corrcoef(scores_for_valid, totals_for_valid)[0, 1]
        else:
            rpb = 0
        
        # Upper/lower 27% discrimination
        n_valid = len(totals_for_valid)
        n27 = max(int(n_valid * 0.27), 1)
        sorted_idx = np.argsort(totals_for_valid)
        lower_mean = np.mean(scores_for_valid[sorted_idx[:n27]])
        upper_mean = np.mean(scores_for_valid[sorted_idx[-n27:]])
        disc_27 = upper_mean - lower_mean
        
        # Get metadata
        meta = metadata.get((assessment_name, str(q)), {})
        
        item_info = {
            'assessment': assessment_name,
            'question_number': q,
            'subject': meta.get('Subject[0].subjectName', ''),
            'grade': meta.get('GradeLevel', ''),
            'content_type': meta.get('ContentSubType', ''),
            'question_text': meta.get('QuestionText', ''),
            'rubric': meta.get('Rubrics[0].criteria', ''),
            'max_score': meta.get('Rubrics[0].score', '1'),
            'skill': meta.get('Skills[0].skillName', ''),
            'n_responses': len(valid),
            'classical_difficulty': round(float(p), 4),
            'item_total_correlation': round(float(rpb), 4),
            'discrimination_27': round(float(disc_27), 4),
            'score_sd': round(float(np.std(valid)), 4),
        }
        all_items.append(item_info)
    
    # Print summary stats
    items_this = [i for i in all_items if i['assessment'] == assessment_name]
    diffs = [i['classical_difficulty'] for i in items_this]
    discs = [i['item_total_correlation'] for i in items_this]
    print(f"  Difficulty range: {min(diffs):.3f} - {max(diffs):.3f} (mean={np.mean(diffs):.3f})")
    print(f"  Item-total corr range: {min(discs):.3f} - {max(discs):.3f} (mean={np.mean(discs):.3f})")
    print()

# Save
outpath = '/Users/dereklomas/AIED/study2-materials/data/smartpaper/item_statistics.json'
with open(outpath, 'w') as f:
    json.dump(all_items, f, indent=2)
print(f"\nSaved {len(all_items)} items to {outpath}")

# Summary table
print("\n=== OVERALL SUMMARY ===")
diffs = [i['classical_difficulty'] for i in all_items]
discs = [i['item_total_correlation'] for i in all_items]
print(f"Total items: {len(all_items)}")
print(f"Difficulty: mean={np.mean(diffs):.3f}, sd={np.std(diffs):.3f}, range=[{min(diffs):.3f}, {max(diffs):.3f}]")
print(f"Item-total r: mean={np.mean(discs):.3f}, sd={np.std(discs):.3f}, range=[{min(discs):.3f}, {max(discs):.3f}]")
print(f"Items with r < 0.2 (poor discrimination): {sum(1 for d in discs if d < 0.2)}")
print(f"Items with p < 0.2 (very hard): {sum(1 for d in diffs if d < 0.2)}")
print(f"Items with p > 0.8 (very easy): {sum(1 for d in diffs if d > 0.8)}")
