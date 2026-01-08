import requests
import os
import time
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from collections import defaultdict
from typing import Optional, Tuple
import time
import re

load_dotenv()

GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
    os.getenv("GROQ_API_KEY_5"),
    os.getenv("GROQ_API_KEY_6"),
    os.getenv("GROQ_API_KEY_7"),
    os.getenv("GROQ_API_KEY_8"),
    os.getenv("GROQ_API_KEY_9"),
    os.getenv("GROQ_API_KEY_10"),
    os.getenv("GROQ_API_KEY_11"),
    os.getenv("GROQ_API_KEY_12")
]
GROQ_API_KEYS = [key for key in GROQ_API_KEYS if key]

if not GROQ_API_KEYS:
    raise ValueError("No GROQ API keys found!")

GROQ_MODEL = "llama-3.3-70b-versatile"

current_key_index = 0
GROQ_RATE_LIMIT_UNTIL = [0] * len(GROQ_API_KEYS)

MAX_TOTAL_TOKENS = 6000
MAX_OUTPUT_TOKENS = 1000
MIN_CHUNK_TOKENS = 200
MAX_CONTEXT_TOKENS = 4000

TEXT_CHUNKS_PER_ITERATION = 2
TABLE_CHUNKS_PER_ITERATION = 2

def get_next_available_key() -> Tuple[Optional[str], int]:
    global current_key_index
    now = time.time()
    
    for _ in range(len(GROQ_API_KEYS)):
        if now >= GROQ_RATE_LIMIT_UNTIL[current_key_index]:
            key = GROQ_API_KEYS[current_key_index]
            index = current_key_index
            current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
            return key, index
        
        current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
    
    earliest_available = min(GROQ_RATE_LIMIT_UNTIL)
    wait_seconds = max(1, int(earliest_available - now))
    return None, wait_seconds

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def compress_chat_history(chat_history, max_items=2):
    if not chat_history:
        return ""
    recent_pairs = []
    for i in range(len(chat_history) - 1, -1, -1):
        if chat_history[i]["role"] == "user":
            user_msg = chat_history[i]["content"]
            if i + 1 < len(chat_history) and chat_history[i + 1]["role"] == "assistant":
                assistant_msg = chat_history[i + 1]["content"]
                recent_pairs.insert(0, (user_msg, assistant_msg))
                if len(recent_pairs) >= max_items:
                    break
    if recent_pairs:
        summary = ["=== Previous Conversation ==="]
        for idx, (q, a) in enumerate(recent_pairs, 1):
            summary.append(f"\nQ{idx}: {q}")
            if len(a) > 300:
                summary.append(f"A{idx}: {a[:300]}...")
            else:
                summary.append(f"A{idx}: {a}")
        summary.append("\n=== End ===\n")
        return "\n".join(summary)
    return ""

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )

def get_system_prompt(language):
    return """You are an accurate assistant for the Master Biomedical Engineering (MBE) program.

GENERAL RULES:
- Use ONLY information explicitly stated in the provided documents or conversation history.
- Do NOT assume, infer, extend, or fabricate information.
- Do NOT use external knowledge.
- Do NOT contradict yourself.

QUESTION HANDLING:
- If the user question contains multiple parts, you MUST internally split it into sub-questions.
- Sub-questions MUST be ONLY the core requirements explicitly stated by the user.
- You MUST NOT invent, expand, or add related or implied questions.
- You MUST answer ONLY what was explicitly asked.

ANSWER RULES:
- Write clear, structured answers using bullet points or short paragraphs.
- Include ONLY factual statements that are directly supported by the sources.
- Do NOT explain reasoning.
- Do NOT add commentary, assumptions, or extra context.
- Do NOT mention document names, page numbers, or sources inside the Answer section.

SOURCE USAGE RULES (STRICT):
- The Sources section MUST include ONLY the sources that were DIRECTLY USED to ASSERT factual statements in the Answer.
- Sources MUST correspond ONLY to the sub-questions that were actually answered.
- If the answer status is Partial information:
  - List ONLY the sources used for the answered sub-questions.
  - Do NOT list sources related to unanswered sub-questions.
- Do NOT list sources that were only checked, reviewed, or inspected.
- Do NOT list sources used only to confirm that information is missing.
- Do NOT list unused, related, or background sources.

STATUS LOGIC (STRICT):
- Internally evaluate EACH sub-question separately.

Status definitions:
- Complete information:
  ALL sub-questions are answered.
- Partial information:
  At least ONE sub-question is answered, but not all.
- No information:
  NONE of the sub-questions are answered.

Hard constraints:
- If ANY sub-question is answered, "No information" is NOT allowed.
- "Complete information" is allowed ONLY if ALL sub-questions are answered.
- Do NOT downgrade status because of missing details outside the asked sub-questions.

OUTPUT FORMAT (EXACT ORDER):

Answer:
<final answer text>

Sources:
- <document name> p<page number>
- (list ONLY sources that directly support the Answer)

Status:
- Write EXACTLY ONE:
  No information
  Partial information
  Complete information

"""

def check_if_answer_insufficient(answer: str) -> bool:
    answer_lower = answer.lower()

    if "❌ no sufficient information found" in answer_lower:
        return True

    if "status:" in answer_lower and "no information" in answer_lower:
        return True

    return False

def check_if_answer_incomplete(answer: str) -> bool:
    answer_lower = answer.lower()
    
    if "status:" in answer_lower and "partial information" in answer_lower:
        return True

    return False

def extract_used_sources_from_answer(answer: str, used_chunks: list) -> list:
    actually_used = []

    match = re.search(r"Sources:\s*(.*)", answer, re.DOTALL | re.IGNORECASE)
    if not match:
        return []

    sources_text = match.group(1).lower()

    source_lines = [
        line.strip("-• ").strip()
        for line in sources_text.splitlines()
        if line.strip()
    ]

    for chunk in used_chunks:
        if not isinstance(chunk, dict):
            continue

        source = chunk.get("source") or chunk.get("metadata", {}).get("source", "")
        page = chunk.get("page") or chunk.get("metadata", {}).get("page", "")

        if not source or not page:
            continue

        source_name = source.split("/")[-1].replace(".pdf", "").lower()

        for line in source_lines:
            if source_name in line and str(page) in line:
                actually_used.append(chunk)
                break

    return actually_used

def get_surrounding_pages_smart(collection, cited_chunks: list, pages_range: int = 1) -> list:
    surrounding_chunks = []
    seen = set()
    
    for chunk in cited_chunks:
        source = chunk.get("source", "")
        page = chunk.get("page", "")
        
        if isinstance(page, str) and "-" in str(page):
            try:
                pages = [int(p) for p in str(page).split("-")]
                current_pages = list(range(pages[0], pages[-1] + 1))
            except:
                try:
                    current_pages = [int(page)]
                except:
                    continue
        else:
            try:
                current_pages = [int(page)]
            except:
                continue
        
        for current_page in current_pages:
            for offset in range(-pages_range, pages_range + 1):
                if offset == 0:
                    continue
                    
                target_page = current_page + offset
                
                if target_page < 1:
                    continue
                
                key = f"{source}_{target_page}"
                if key in seen:
                    continue
                seen.add(key)
                
                try:
                    results = collection.query(
                        query_texts=["context retrieval"],
                        n_results=5,
                        where={
                            "$and": [
                                {"source": source},
                                {"page": target_page}
                            ]
                        }
                    )
                    
                    if results["documents"][0]:
                        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                            surrounding_chunks.append({
                                "content": doc,
                                "metadata": meta,
                                "source": meta.get("source", "Unknown"),
                                "page": meta.get("page", "N/A"),
                                "type": meta.get("type", "text")
                            })
                except Exception as e:
                    continue
    
    return surrounding_chunks

def get_chunks_from_same_sources(collection, cited_sources: list, already_used_pages: set, n_per_source: int = 10) -> list:
    new_chunks = []
    seen = set()
        
    for chunk in cited_sources:
        source = chunk.get("source", "")
        
        if not source:
            continue
        
        try:
            results = collection.query(
                query_texts=["additional context"],
                n_results=n_per_source * 2,
                where={"source": source}
            )
            
            if results["documents"][0]:
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    page = meta.get("page", "N/A")
                    key = f"{source}_{page}"
                    
                    if key in seen or key in already_used_pages:
                        continue
                    
                    seen.add(key)
                    new_chunks.append({
                        "content": doc,
                        "metadata": meta,
                        "source": meta.get("source", "Unknown"),
                        "page": page,
                        "type": meta.get("type", "text")
                    })
                    
                    if len(new_chunks) >= n_per_source:
                        break
        except Exception as e:
            continue
    
    return new_chunks

def search_chunks_simple(collection, query: str, n_text: int = 80, n_tables: int = 40):
    text_chunks = []
    table_chunks = []
    
    try:
        table_results = collection.query(
            query_texts=[query],
            n_results=n_tables,
            where={"type": {"$in": ["table_with_context", "table"]}}
        )
        
        for doc, meta in zip(table_results["documents"][0], table_results["metadatas"][0]):
            table_chunks.append({
                "content": doc,
                "metadata": meta,
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", "N/A"),
                "type": meta.get("type", "table")
            })
    except Exception as e:
        pass
    
    try:
        text_results = collection.query(
            query_texts=[query],
            n_results=n_text,
            where={"type": {"$nin": ["table_with_context", "table"]}}
        )
        
        for doc, meta in zip(text_results["documents"][0], text_results["metadatas"][0]):
            chunk_tokens = estimate_tokens(doc)
            if chunk_tokens >= MIN_CHUNK_TOKENS:
                text_chunks.append({
                    "content": doc,
                    "metadata": meta,
                    "source": meta.get("source", "Unknown"),
                    "page": meta.get("page", "N/A"),
                    "type": meta.get("type", "text")
                })
    except Exception as e:
        pass
    
    return text_chunks, table_chunks

def trim_context_to_fit(context_parts: list, max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    total_text = "\n\n---\n\n".join(context_parts)
    current_tokens = estimate_tokens(total_text)
    
    if current_tokens <= max_tokens:
        return total_text
    
    while current_tokens > max_tokens and len(context_parts) > 1:
        context_parts.pop()
        total_text = "\n\n---\n\n".join(context_parts)
        current_tokens = estimate_tokens(total_text)
    
    return total_text

def call_groq_model(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.05,
    max_tokens: int = MAX_OUTPUT_TOKENS
) -> Tuple[str, bool]:

    while True:
        api_key, info = get_next_available_key()

        if api_key is None:
            return (f"All API keys are rate limited. Please wait {info} seconds.", []) 

        data = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=data,
                timeout=60
            )

            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"].strip()
            return answer, True

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after else 60
                GROQ_RATE_LIMIT_UNTIL[info] = time.time() + wait_time
                continue 

            elif e.response is not None and e.response.status_code == 413:
                return "Payload too large", False

            return f"HTTP Error: {str(e)}", False

        except Exception as e:
            time.sleep(5)
            continue

def prepare_iteration_context(
    cumulative_cited_sources: list,
    new_text_batch: list,
    new_table_batch: list,
    max_tokens: int = MAX_CONTEXT_TOKENS,
    max_used_sources: int = 6
) -> Tuple[str, list]:

    context_parts = []
    current_iteration_chunks = []
    seen_keys = set()

    if cumulative_cited_sources:
        used_sources_limited = cumulative_cited_sources[-max_used_sources:]

        for chunk in used_sources_limited:
            source = chunk.get("source", "Unknown")
            page = chunk.get("page", "N/A")
            content = chunk.get("content", "")
            chunk_type = chunk.get("type", "text")

            key = f"{source}_{page}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            type_marker = "[TABLE]" if chunk_type in ["table", "table_with_context"] else "[TEXT]"
            context_parts.append(
                f"[📌 USED {type_marker} {source} p{page}]\n{content}"
            )
            current_iteration_chunks.append(chunk)

    for chunk in new_table_batch:
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "N/A")
        key = f"{source}_{page}"

        if key in seen_keys:
            continue
        seen_keys.add(key)

        context_parts.append(
            f"[📊 NEW {source} p{page}]\n{chunk.get('content', '')}"
        )
        current_iteration_chunks.append(chunk)

    for chunk in new_text_batch:
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "N/A")
        key = f"{source}_{page}"

        if key in seen_keys:
            continue
        seen_keys.add(key)

        context_parts.append(
            f"[📄 NEW {source} p{page}]\n{chunk.get('content', '')}"
        )
        current_iteration_chunks.append(chunk)

    context = trim_context_to_fit(context_parts, max_tokens)

    return context, current_iteration_chunks

def get_next_chunk_batch(
    iteration: int,
    all_text_chunks: list,
    all_table_chunks: list,
    text_index: int,
    table_index: int,
    cumulative_cited_sources: list,
    last_answer: str,
    used_pages: set,
    collection,
    source_expansion_steps: dict
) -> Tuple[list, list, bool]:

    
    if iteration == 1:
        new_text_batch = all_text_chunks[text_index:text_index + TEXT_CHUNKS_PER_ITERATION]
        new_table_batch = all_table_chunks[table_index:table_index + TABLE_CHUNKS_PER_ITERATION]
        return new_text_batch, new_table_batch, False
    
    is_incomplete = check_if_answer_incomplete(last_answer)

    if is_incomplete and cumulative_cited_sources:
        seen_sources = set()
        BASE_SURROUNDING_RANGE = 1
        MAX_SURROUNDING_RANGE = 4

        expanded_chunks = []

        for chunk in cumulative_cited_sources:
            source = chunk.get("source")
            if not source or source in seen_sources:
                continue
            seen_sources.add(source)

            if source not in source_expansion_steps:
                source_expansion_steps[source] = 0

            current_step = source_expansion_steps.get(source, 0)

            dynamic_range = min(
                BASE_SURROUNDING_RANGE + current_step,
                MAX_SURROUNDING_RANGE
            )

            surrounding = get_surrounding_pages_smart(
                collection,
                [chunk],
                pages_range=dynamic_range
            )

            if surrounding:
                source_expansion_steps[source] = current_step + 1
                expanded_chunks.extend(surrounding)

        filtered_chunks = []
        for ch in expanded_chunks:
            key = f"{ch.get('source')}_{ch.get('page')}"
            if key not in used_pages:
                used_pages.add(key)
                filtered_chunks.append(ch)

        new_text_batch = [
            c for c in filtered_chunks
            if c.get("type") not in ["table", "table_with_context"]
        ]

        new_table_batch = [
            c for c in filtered_chunks
            if c.get("type") in ["table", "table_with_context"]
        ]

        if new_text_batch or new_table_batch:
            return new_text_batch, new_table_batch, True
    
    if text_index >= len(all_text_chunks) and table_index >= len(all_table_chunks):
       return [], [], False
    
    new_text_batch = []
    new_table_batch = []
    
    candidate_text = all_text_chunks[text_index:text_index + TEXT_CHUNKS_PER_ITERATION]
    for chunk in candidate_text:
        key = f"{chunk.get('source')}_{chunk.get('page')}"
        if key not in used_pages:
            new_text_batch.append(chunk)
    
    candidate_tables = all_table_chunks[table_index:table_index + TABLE_CHUNKS_PER_ITERATION]
    for chunk in candidate_tables:
        key = f"{chunk.get('source')}_{chunk.get('page')}"
        if key not in used_pages:
            new_table_batch.append(chunk)
    
    return new_text_batch, new_table_batch, False

def process_iteration_result(
    answer: str,
    current_iteration_chunks: list,
    cumulative_cited_sources: list,
    used_pages: set,
    iteration: int
) -> Tuple[list, bool]:
    
    is_insufficient = check_if_answer_insufficient(answer)
    is_incomplete = check_if_answer_incomplete(answer)
    if is_insufficient:
        cited_in_this_iteration = []
    else:
        cited_in_this_iteration = extract_used_sources_from_answer(
            answer, current_iteration_chunks
        )

    updated_cumulative = cumulative_cited_sources.copy()
    for new_source in cited_in_this_iteration:
        is_duplicate = False
        for existing in updated_cumulative:
            if (existing.get("source") == new_source.get("source") and 
                existing.get("page") == new_source.get("page")):
                is_duplicate = True
                break
        if not is_duplicate:
            updated_cumulative.append(new_source)
            key = f"{new_source.get('source')}_{new_source.get('page')}"
            used_pages.add(key)

    is_complete = (
        (not is_insufficient) and 
        (not is_incomplete)
    )
    
    return updated_cumulative, is_complete

def answer_question_with_groq(query, chat_history=None, user_language="en", collection=None, ):
    
    if not collection:
        return "Collection is required", []
    all_text_chunks, all_table_chunks = search_chunks_simple(
        collection, query, n_text=80, n_tables=40
    )
    
    if not all_text_chunks and not all_table_chunks:
        return "❌ No information available in the documents.", []
    
    conversation_summary = compress_chat_history(chat_history, max_items=2)
    system_prompt = get_system_prompt(user_language)
    
    cumulative_cited_sources = []
    iteration = 1
    text_index = 0
    table_index = 0
    last_answer = ""

    used_pages = set()
    source_expansion_steps = {}

    while True:
        
        new_text_batch, new_table_batch, is_expanding = get_next_chunk_batch(
                iteration,
                all_text_chunks,
                all_table_chunks,
                text_index,
                table_index,
                cumulative_cited_sources,
                last_answer,
                used_pages,
                collection,
                source_expansion_steps
            )

        
        if not new_text_batch and not new_table_batch:
            break
        
        context, current_iteration_chunks = prepare_iteration_context(
            cumulative_cited_sources, new_text_batch, new_table_batch, MAX_CONTEXT_TOKENS
        )
        
        user_content = f"""{conversation_summary if conversation_summary else ''}

SOURCES:
{context}
QUESTION: {query}
Answer strictly according to the system rules. Follow the required output format exactly.
"""
        answer, success = call_groq_model(system_prompt, user_content)
        if not success:
            if "Rate limited" in answer:
                return answer, []
            elif "Payload too large" in answer:
                continue
            else:
                return answer, []
        
        last_answer = answer
        
        cumulative_cited_sources, is_complete = \
                process_iteration_result(
                    answer,
                    current_iteration_chunks,
                    cumulative_cited_sources,
                    used_pages,
                    iteration
                )
        if is_complete:
            api_key, _ = get_next_available_key()
            if api_key:
                final_answer = answer 
            return final_answer, cumulative_cited_sources
        
        if not is_expanding:
            text_index += TEXT_CHUNKS_PER_ITERATION
            table_index += TABLE_CHUNKS_PER_ITERATION
        
        iteration += 1
    
    return last_answer, cumulative_cited_sources