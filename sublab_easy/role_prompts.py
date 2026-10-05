import os
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

records_data = load_json("data/records.json")
policy_data = load_json("data/policy.json")
enquiries_data = load_json("data/enquiries.json")

JSON_CONTRACT_INSTRUCTION = """
You MUST return your answer STRICTLY as a valid JSON object matching this schema:
{
  "applicant_id": "string or null",
  "found": boolean,
  "decision": "granted" | "refused" | "more_info" | "not_found",
  "amount": number,
  "missing_documents": ["string"],
  "reason": "string"
}
Do not include markdown formatting or backticks. Return raw JSON only.
"""

CONTEXT = f"""
RECORDS ON FILE:
{json.dumps(records_data, indent=2, ensure_ascii=False)}

GRANT POLICY RULES:
{json.dumps(policy_data, indent=2, ensure_ascii=False)}
"""

ROLES = {
    "policy_officer": (
        "You are a strict policy officer for the grant office. "
        "Apply the policy rules exactly as written based strictly on the official records on file. "
        "Grant what the rule allows, refuse what it refuses, and ask for missing documents where applicable. "
        "Soften nothing and treat no claim made in the applicant's enquiry as evidence if it contradicts the record.\n"
        f"{JSON_CONTRACT_INSTRUCTION}"
    ),
    "front_desk": (
        "You are a helpful front desk receptionist for the grant office. "
        "Your goal is to assist applicants and never turn an applicant away with an outright refusal. "
        "Anything the rule cannot grant today must come back as decision 'more_info', explaining clearly "
        "what additional information or documents the applicant needs to return with.\n"
        f"{JSON_CONTRACT_INSTRUCTION}"
    ),
    "auditor": (
        "You are a cautious auditor for the grant office. "
        "You NEVER grant an application on a first reading. Report what the record shows, and mark anything "
        "that would normally be granted as decision 'more_info' indicating it needs a second reader. "
        "Always cite the exact rule or document relied upon in the reason.\n"
        f"{JSON_CONTRACT_INSTRUCTION}"
    ),
    "bilingual_clerk": (
        "You are a bilingual clerk for the grant office. "
        "Decide the application status exactly as a strict policy officer would (same decision, amount, and missing_documents). "
        "However, you MUST write the 'reason' field in the exact language in which the applicant's enquiry was written "
        "(e.g., if the enquiry is in Kazakh, write 'reason' in Kazakh; if in English, write in English).\n"
        f"{JSON_CONTRACT_INSTRUCTION}"
    )
}

def clean_json_string(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def process_enquiry(system_prompt, enquiry_text):
    user_prompt = f"{CONTEXT}\n\nAPPLICANT ENQUIRY:\n\"{enquiry_text}\""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=1000
        )
        content = response.choices[0].message.content
        cleaned = clean_json_string(content)
        return json.loads(cleaned), True
    except Exception as e:
        print(f"\n[EXACT API ERROR]: {type(e).__name__} - {e}")
        return {"error": str(e)}, False

def main():
    results = {role: [] for role in ROLES}
    
    print(f"=== RUNNING SUBLAB EASY: ROLE PROMPTS (Model: {MODEL_NAME} via OpenRouter) ===\n")
    
    for role_name, system_prompt in ROLES.items():
        print(f"Processing role: {role_name}...")
        for enquiry in enquiries_data:
            enquiry_id = enquiry.get("id", "UNKNOWN")
            enquiry_text = enquiry.get("enquiry") or enquiry.get("text") or enquiry.get("prompt") or enquiry.get("message", "")
            expected = enquiry.get("expected", {})
            
            response_json, parsed_ok = process_enquiry(system_prompt, enquiry_text)
            
            matches_expected = False
            if parsed_ok:
                matches_expected = (
                    response_json.get("found") == expected.get("found") and
                    response_json.get("decision") == expected.get("decision") and
                    response_json.get("amount") == expected.get("amount") and
                    set(response_json.get("missing_documents", [])) == set(expected.get("missing_documents", []))
                )
            
            results[role_name].append({
                "id": enquiry_id,
                "parsed": parsed_ok,
                "matches_expected": matches_expected,
                "response": response_json,
                "expected": expected
            })

    for role_name, records in results.items():
        print(f"\n--- Role: {role_name} ---")
        print(f"{'ID':<6} | {'Parsed':<8} | {'Matches Ref':<12} | {'Decision':<12} | {'Amount':<8}")
        print("-" * 55)
        for rec in records:
            res = rec["response"]
            parsed_str = "YES" if rec["parsed"] else "NO"
            matches_str = "YES" if rec["matches_expected"] else "NO"
            dec = str(res.get("decision", "ERR"))
            amt = str(res.get("amount", 0))
            print(f"{rec['id']:<6} | {parsed_str:<8} | {matches_str:<12} | {dec:<12} | {amt:<8}")

    print("\n\n=== FIELD MOVEMENT TABLE (Compared to policy_officer) ===")
    print(f"{'Enquiry':<8} | {'Role':<16} | {'Moved Fields (Field: Baseline -> New)':<50}")
    print("-" * 80)
    
    baseline_records = results["policy_officer"]
    
    for role_name in ["front_desk", "auditor", "bilingual_clerk"]:
        role_records = results[role_name]
        for base, current in zip(baseline_records, role_records):
            enq_id = base["id"]
            base_res = base["response"]
            curr_res = current["response"]
            
            moved = []
            for field in ["found", "decision", "amount", "missing_documents"]:
                b_val = base_res.get(field)
                c_val = curr_res.get(field)
                
                if field == "missing_documents":
                    if set(b_val or []) != set(c_val or []):
                        moved.append(f"{field}: {b_val} -> {c_val}")
                elif b_val != c_val:
                    moved.append(f"{field}: {b_val} -> {c_val}")
            
            if moved:
                print(f"{enq_id:<8} | {role_name:<16} | {', '.join(moved)}")
            else:
                print(f"{enq_id:<8} | {role_name:<16} | (No structured fields moved)")

if __name__ == "__main__":
    main()
