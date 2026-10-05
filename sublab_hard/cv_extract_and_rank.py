import os
import glob
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.6-luna")

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

rubric_data = load_json("data/candidate_rubric.json")

def clean_json_string(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# PART 1: EXTRACTION PROMPT WITH STRICT TRAP RULES
EXTRACTION_SYSTEM_PROMPT = """
You are a precise data extraction specialist for university scholarship applications.
Extract a structured CV record from the candidate's story.

STRICT EXTRACTION RULES:
1. Missing Information: A fact that the story does not explicitly state MUST be null. Never estimate, infer, or guess.
2. GPA & Scale:
   - Convert any GPA to a 4.0 scale (`gpa_4_scale`). Record the original scale in `original_gpa_scale` (e.g., "5.0", "100", "4.0").
   - If no GPA is mentioned, `gpa_4_scale` MUST be null.
3. Publication Counting:
   - Count a paper in `published_peer_reviewed_count` ONLY if the story explicitly states it is 'published' or 'accepted'.
   - Papers described as 'submitted', 'under review', 'in preparation', 'planned', or 'in press' are NOT published. Record them in `unpublished_or_pending_count` and do NOT count them towards published papers.
4. Contradictions:
   - If the story contradicts itself regarding a value (e.g. mentions two different GPAs or conflicting dates), do NOT average or resolve it. Set that field to null and record the details in `contradictions`.
5. Kazakh language: If the story is in Kazakh, read it accurately and return all JSON fields in English.
6. Evidence Quotes: Provide exact evidence quotes from the story for each extracted field.

OUTPUT JSON SCHEMA:
{
  "candidate_id": "string",
  "full_name": "string or null",
  "degree": "string or null",
  "graduation_year": "number or null",
  "gpa_4_scale": "number or null",
  "original_gpa_scale": "string or null",
  "languages": ["string"],
  "published_peer_reviewed_count": "number",
  "unpublished_or_pending_count": "number",
  "total_countable_months_experience": "number",
  "contradictions": ["string"],
  "evidence_quotes": {
    "gpa_evidence": "string or null",
    "publication_evidence": "string or null",
    "experience_evidence": "string or null"
  }
}
Return raw JSON only. Do not wrap in markdown or code blocks.
"""

# PART 2: SCORING PROMPT
SCORING_SYSTEM_PROMPT = f"""
You are an academic reviewer evaluating candidate CVs against a specific rubric.
Evaluate the extracted candidate record and assign scores strictly from 0 to 5 based on these rules:

RUBRIC RULES:
{json.dumps(rubric_data, indent=2, ensure_ascii=False)}

OUTPUT FORMAT:
You must return ONLY a JSON object containing the scores (0 to 5) for the three criteria:
{{
  "academic": number,
  "research": number,
  "experience": number
}}
Do NOT compute the total score or name the winner. Return scores only.
"""

def extract_cv(candidate_id, story_text):
    user_prompt = f"CANDIDATE ID: {candidate_id}\n\nCANDIDATE STORY:\n{story_text}"
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=1000
    )
    content = clean_json_string(response.choices[0].message.content)
    return json.loads(content)

def score_cv(cv_record):
    user_prompt = f"EXTRACTED CANDIDATE CV RECORD:\n{json.dumps(cv_record, indent=2, ensure_ascii=False)}"
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SCORING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=500
    )
    content = clean_json_string(response.choices[0].message.content)
    return json.loads(content)

def get_prose_winner(all_stories):
    combined_stories = "\n\n---\n\n".join([f"Story ID: {sid}\nText:\n{stext}" for sid, stext in all_stories.items()])
    prompt = f"Here are six candidate stories for a funded scholarship:\n\n{combined_stories}\n\nBased on academic strength, research outputs, and relevant experience, write a short prose recommendation naming which candidate should win and explaining why."
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800
    )
    return response.choices[0].message.content

def main():
    print("=== SUBLAB HARD: CV EXTRACTION AND RANKING ===\n")
    story_files = sorted(glob.glob("data/candidates/story-*.md"))
    
    extracted_records = []
    scores_dict = {}
    all_stories = {}

    print("Step 1: Extracting CVs...")
    for file_path in story_files:
        candidate_id = os.path.basename(file_path).replace(".md", "")
        with open(file_path, "r", encoding="utf-8") as f:
            story_text = f.read()
        all_stories[candidate_id] = story_text
        
        cv = extract_cv(candidate_id, story_text)
        extracted_records.append(cv)
        print(f" Extracted {candidate_id}: Name={cv.get('full_name')}, GPA={cv.get('gpa_4_scale')}, Pubs={cv.get('published_peer_reviewed_count')}, ExpMonths={cv.get('total_countable_months_experience')}")

    print("\nStep 2: Scoring Candidates with LLM...")
    for cv in extracted_records:
        cid = cv["candidate_id"]
        sc = score_cv(cv)
        scores_dict[cid] = sc
        print(f" Scores for {cid}: Academic={sc['academic']}, Research={sc['research']}, Experience={sc['experience']}")

    print("\nStep 3: Computing Weighted Total in Code...")
    ranked_candidates = []
    for cv in extracted_records:
        cid = cv["candidate_id"]
        sc = scores_dict[cid]
        # Formula: 0.5 * academic + 0.3 * research + 0.2 * experience
        weighted_total = round(0.5 * sc["academic"] + 0.3 * sc["research"] + 0.2 * sc["experience"], 2)
        ranked_candidates.append({
            "candidate_id": cid,
            "name": cv.get("full_name"),
            "scores": sc,
            "weighted_total": weighted_total,
            "cv": cv
        })

    ranked_candidates.sort(key=lambda x: x["weighted_total"], reverse=True)

    print("\n" + "="*70)
    print("COMPUTED RANKING TABLE")
    print("="*70)
    print(f"{'Rank':<5} | {'ID':<10} | {'Name':<20} | {'Acad':<6} | {'Res':<6} | {'Exp':<6} | {'Total':<6}")
    print("-" * 70)
    for idx, cand in enumerate(ranked_candidates, 1):
        s = cand["scores"]
        print(f"{idx:<5} | {cand['candidate_id']:<10} | {str(cand['name']):<20} | {s['academic']:<6} | {s['research']:<6} | {s['experience']:<6} | {cand['weighted_total']:<6}")

    winner = ranked_candidates[0]
    print(f"\nCOMPUTED WINNER (BY CODE): {winner['candidate_id']} ({winner['name']}) with Score = {winner['weighted_total']}")

    print("\nStep 4: Requesting Prose Recommendation from LLM...")
    prose_recommendation = get_prose_winner(all_stories)
    print("\n--- LLM PROSE RECOMMENDATION ---")
    print(prose_recommendation)

    # Save full results for inspection
    output_payload = {
        "ranked_candidates": ranked_candidates,
        "prose_recommendation": prose_recommendation
    }
    with open("sublab_hard_results.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    print("\n[SUCCESS] Saved detailed results to sublab_hard_results.json")

if __name__ == "__main__":
    main()
