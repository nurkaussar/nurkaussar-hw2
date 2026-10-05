import os
import sys
import json
import argparse
from dotenv import load_dotenv
from openai import OpenAI
import jsonschema

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

schema = load_json("data/memory_state.schema.json")
script_data = load_json("data/chat_script.json")

SYSTEM_PROMPT = "You are a helpful grant office assistant. Assist applicants clearly and concisely based on past context."

COMPRESSION_PROMPT = f"""
Summarize the current conversation into a structured JSON state object.
You MUST output STRICTLY a JSON object matching this schema:
{json.dumps(schema, indent=2)}

Do not include backticks, markdown formatting, or any extra text. Return valid raw JSON only.
"""

def validate_state(state_obj):
    try:
        jsonschema.validate(instance=state_obj, schema=schema)
        return True, ""
    except jsonschema.exceptions.ValidationError as err:
        return False, err.message

def clean_json_string(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def run_scripted_session(use_compression=True):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    current_state = None
    token_log = []
    probe_results = {}

    turns = script_data.get("script", [])
    probes = script_data.get("probes", [])

    print(f"\n--- RUNNING SESSION (Compressed: {use_compression}) ---")

    for idx, turn in enumerate(turns):
        user_text = turn.get("text", turn.get("user", ""))
        
        # Перехват команды compress
        if user_text.strip() == "compress":
            if use_compression:
                # Запрос на сжатие контекста
                comp_messages = messages + [{"role": "user", "content": COMPRESSION_PROMPT}]
                try:
                    res = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=comp_messages,
                        temperature=0.0,
                        max_tokens=1000
                    )
                    raw_summary = res.choices[0].message.content
                    cleaned = clean_json_string(raw_summary)
                    state_obj = json.loads(cleaned)
                    
                    is_valid, err_msg = validate_state(state_obj)
                    if is_valid:
                        current_state = state_obj
                        # Сбрасываем историю сообщений и оставляем только сжатый State
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "system", "content": f"CONVERSATION STATE / MEMORY:\n{json.dumps(current_state, ensure_ascii=False)}"}
                        ]
                        print(f" Turn {idx+1} [compress]: Successfully compressed into valid state schema.")
                    else:
                        print(f" Turn {idx+1} [compress]: Schema validation failed ({err_msg}). Retaining history.")
                except Exception as e:
                    print(f" Turn {idx+1} [compress]: Error during compression ({e}). Retaining history.")
            else:
                print(f" Turn {idx+1} [compress]: Skipped compression (Uncompressed mode).")
            continue

        # Обычный шаг диалога
        messages.append({"role": "user", "content": user_text})
        
        try:
            res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=1000
            )
            reply = res.choices[0].message.content
            tokens_sent = res.usage.prompt_tokens if hasattr(res, 'usage') and res.usage else len(str(messages)) // 4
            token_log.append((idx + 1, user_text[:30], tokens_sent))
            
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f" Error on turn {idx+1}: {e}")

    # Проверка проверочных вопросов (Probes)
    print("\nEvaluating Probes:")
    for probe in probes:
        probe_id = probe.get("id", "P-UNKNOWN")
        question = probe.get("question", probe.get("text", ""))
        expected_fact = probe.get("expected_fact", "")
        
        probe_messages = messages + [{"role": "user", "content": question}]
        try:
            res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=probe_messages,
                temperature=0.0,
                max_tokens=500
            )
            answer = res.choices[0].message.content
            retained = expected_fact.lower() in answer.lower() if expected_fact else True
            probe_results[probe_id] = {
                "question": question,
                "answer": answer,
                "retained": retained
            }
            print(f" [{probe_id}] Retained: {'YES' if retained else 'NO'} | Q: {question}")
        except Exception as e:
            probe_results[probe_id] = {"error": str(e), "retained": False}

    return token_log, probe_results, current_state

def interactive_mode():
    print("=== INTERACTIVE CHAT MEMORY MODE ===")
    print("Type your message. Type 'compress' to trigger compression. Type 'exit' to quit.\n")
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    current_state = None

    while True:
        user_input = input("User > ").strip()
        if user_input.lower() == "exit":
            break
        
        if user_input.lower() == "compress":
            comp_messages = messages + [{"role": "user", "content": COMPRESSION_PROMPT}]
            try:
                res = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=comp_messages,
                    temperature=0.0,
                    max_tokens=1000
                )
                raw_summary = res.choices[0].message.content
                cleaned = clean_json_string(raw_summary)
                state_obj = json.loads(cleaned)
                
                is_valid, err_msg = validate_state(state_obj)
                if is_valid:
                    current_state = state_obj
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "system", "content": f"CONVERSATION STATE / MEMORY:\n{json.dumps(current_state, indent=2, ensure_ascii=False)}"}
                    ]
                    print("\n[SYSTEM]: Compression successful! State object updated:")
                    print(json.dumps(current_state, indent=2, ensure_ascii=False))
                else:
                    print(f"\n[SYSTEM]: Compression failed validation: {err_msg}")
            except Exception as e:
                print(f"\n[SYSTEM]: Compression error: {e}")
            continue

        messages.append({"role": "user", "content": user_input})
        try:
            res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=1000
            )
            reply = res.choices[0].message.content
            tokens_used = res.usage.prompt_tokens if hasattr(res, 'usage') and res.usage else 'N/A'
            print(f"Assistant > {reply}")
            print(f"--- [Tokens sent in last call: {tokens_used}] ---\n")
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"Error: {e}\n")

def main():
    parser = argparse.ArgumentParser(description="Sublab Medium: Chat Memory Compression")
    parser.add_argument("--interactive", action="store_true", help="Run interactive chat mode")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    else:
        # Запуск двух прогонов: Uncompressed и Compressed
        uncompressed_tokens, uncompressed_probes, _ = run_scripted_session(use_compression=False)
        compressed_tokens, compressed_probes, state_obj = run_scripted_session(use_compression=True)

        print("\n" + "="*60)
        print("SUMMARY RESULTS FOR SUBMISSION.md")
        print("="*60)

        print("\n--- TOKEN COMPARISON TABLE ---")
        print(f"{'Turn':<6} | {'User Text Snippet':<30} | {'Uncompressed Tokens':<20} | {'Compressed Tokens':<18}")
        print("-" * 80)
        
        max_u = max([t[2] for t in uncompressed_tokens]) if uncompressed_tokens else 0
        max_c = max([t[2] for t in compressed_tokens]) if compressed_tokens else 0

        for u, c in zip(uncompressed_tokens, compressed_tokens):
            print(f"{u[0]:<6} | {u[1]:<30} | {u[2]:<20} | {c[2]:<18}")
        
        print("-" * 80)
        print(f"PEAK TOKENS: Uncompressed = {max_u} | Compressed = {max_c}\n")

        print("--- PROBE RETRIEVAL RESULTS ---")
        for pid in uncompressed_probes:
            u_ret = "YES" if uncompressed_probes[pid]["retained"] else "NO"
            c_ret = "YES" if compressed_probes[pid]["retained"] else "NO"
            print(f"Probe {pid}: Uncompressed={u_ret} | Compressed={c_ret}")

        if state_obj:
            print("\n--- COMPRESSED STATE OBJECT ---")
            print(json.dumps(state_obj, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
