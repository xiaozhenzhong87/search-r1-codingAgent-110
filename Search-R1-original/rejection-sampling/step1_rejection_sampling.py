"""
Step 1: Rejection Sampling
For each question in the training set, sample N responses via LLM API,
simulate multi-turn <think>/<search>/<information>/<answer> interaction,
compute QA EM reward, and save all candidates to jsonl.
"""

import re, json, time, string, random, logging, argparse, threading, requests
import pandas as pd, numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# ── Reward: QA Exact Match ───────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t): return " ".join(t.split())
    def remove_punc(t): return "".join(c for c in t if c not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str): golden_answers = [golden_answers]
    norm = normalize_answer(prediction)
    return any(normalize_answer(g) == norm for g in golden_answers)

def compute_reward(full_text, golden_answers, format_score=0.1):
    """Extract <answer> from full text (prompt+response). Need >=2 matches
    because the prompt itself contains an example <answer>."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", full_text, re.DOTALL))
    if len(matches) <= 1: return 0.0
    answer = matches[-1].group(1).strip()
    return 1.0 if em_check(answer, golden_answers) else format_score

def extract_answer_from_response(text):
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    return matches[-1].group(1).strip() if matches else None

# ── LLM API Client ───────────────────────────────────────────────────────

class LLMClient:
    """
    Mediago OpenAI proxy client.
    API format:
      - model_name, context (not messages), text+role_type (not content+role)
      - max_token / ans_token are ints; temperature / top_p are strings
      - response: {"errno":0, "data":{"content":"..."}}
    """

    ROLE_MAP = {"user": "user", "assistant": "assistant", "system": "system"}

    def __init__(self, api_url, headers, model="deploy_gpt5_chat",
                 temperature=1.0, max_tokens=2048,
                 max_retries=5, retry_base_delay=2.0,
                 max_concurrent=2, min_interval=0.5):
        self.api_url = api_url
        self.headers = headers
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._local = threading.local()
        self._semaphore = threading.Semaphore(max_concurrent)
        self._lock = threading.Lock()
        self._last_request_time = 0.0
        self._min_interval = min_interval

    def _get_session(self):
        """Thread-local session to avoid thread-safety issues."""
        if not hasattr(self._local, 'session'):
            self._local.session = requests.Session()
            self._local.session.headers.update(self.headers)
        return self._local.session

    @staticmethod
    def _sanitize(text):
        """Remove control characters (keep newline/tab) to avoid API parse errors."""
        if not text:
            return " "
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    @classmethod
    def _to_context(cls, messages):
        """Convert standard OpenAI messages to mediago context format.
        Ensures every entry has non-empty text."""
        ctx = []
        for m in messages:
            text = cls._sanitize(m.get("content", ""))
            ctx.append({"text": text, "role_type": m["role"]})
        return ctx

    def chat(self, messages, stop=None, temperature=None):
        """Send chat request. messages use standard OpenAI format internally;
        converted to mediago format before sending."""
        temp = temperature if temperature is not None else self.temperature
        context = self._to_context(messages)
        payload = {
            "model_name": self.model,
            "context": context,
            "max_token": self.max_tokens,
            "ans_token": self.max_tokens,
            "temperature": str(temp),
            "top_p": "1.0",
            "frequency_penalty": "0.0",
            "presence_penalty": "0.0",
            "api_version": "2023-05-15",
        }
        if stop:
            payload["stop"] = stop

        session = self._get_session()
        for attempt in range(1, self.max_retries + 1):
            data = None
            with self._semaphore:
                with self._lock:
                    elapsed = time.time() - self._last_request_time
                    if elapsed < self._min_interval:
                        time.sleep(self._min_interval - elapsed + random.random() * 0.2)
                    self._last_request_time = time.time()
                try:
                    resp = session.post(self.api_url, json=payload, timeout=120)
                    data = resp.json()
                except Exception as e:
                    if attempt < self.max_retries:
                        delay = self.retry_base_delay * (2 ** (attempt - 1)) + random.random()
                        logger.warning("API exception (attempt %d/%d): %s - retry %.1fs",
                                       attempt, self.max_retries, e, delay)
                        time.sleep(delay)
                        continue
                    logger.error("API exception after %d retries: %s", self.max_retries, e)
                    return "", "error"

            if data is None:
                continue

            errno = data.get("errno", -1)
            if errno != 0:
                msg = data.get("msg", "unknown")
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** (attempt - 1)) + random.random()
                    logger.warning("API error (attempt %d/%d errno=%d): %s - retry %.1fs",
                                   attempt, self.max_retries, errno, msg, delay)
                    time.sleep(delay)
                    continue
                logger.error("API error after %d retries: %s", self.max_retries, msg)
                return "", "error"

            d = data.get("data", {})
            if isinstance(d, dict):
                return d.get("content", ""), "stop"

            if "choices" in data:
                ch = data["choices"][0]
                return ch.get("message", {}).get("content", ""), ch.get("finish_reason", "stop")

            logger.warning("Unrecognized response: %s", json.dumps(data, ensure_ascii=False)[:200])
            return str(d), "stop"

        logger.error("API failed after %d retries", self.max_retries)
        return "", "error"


class VLLMClient:
    """Client for vLLM OpenAI-compatible API (completions endpoint).
    Uses /v1/completions (not chat) since the RL model is prompt-based."""

    def __init__(self, api_url, model="qwen-7b-rl",
                 temperature=1.0, max_tokens=2048,
                 max_retries=5, retry_base_delay=1.0,
                 max_concurrent=8, api_key="EMPTY"):
        self.base_url = api_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._local = threading.local()
        self._semaphore = threading.Semaphore(max_concurrent)
        self.api_key = api_key

    def _get_session(self):
        if not hasattr(self._local, 'session'):
            self._local.session = requests.Session()
            self._local.session.headers.update({
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            })
        return self._local.session

    def chat(self, messages, stop=None, temperature=None):
        """For compatibility with multi-turn generation interface.
        Converts messages to a single prompt string for completions API."""
        prompt = self._messages_to_prompt(messages)
        return self.complete(prompt, stop=stop, temperature=temperature)

    @staticmethod
    def _messages_to_prompt(messages):
        """Concatenate messages into a flat prompt.
        For the RL model, the first user message is the full prompt,
        and subsequent turns are assistant/user alternation."""
        parts = []
        for m in messages:
            parts.append(m["content"])
        return "".join(parts)

    def complete(self, prompt, stop=None, temperature=None):
        """Send a completions request to vLLM."""
        temp = temperature if temperature is not None else self.temperature
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": temp,
            "top_p": 1.0,
        }
        if stop:
            payload["stop"] = stop

        url = f"{self.base_url}/v1/completions"
        session = self._get_session()

        for attempt in range(1, self.max_retries + 1):
            with self._semaphore:
                try:
                    resp = session.post(url, json=payload, timeout=180)
                    data = resp.json()
                except Exception as e:
                    if attempt < self.max_retries:
                        delay = self.retry_base_delay * (2 ** (attempt - 1)) + random.random()
                        logger.warning("vLLM exception (attempt %d/%d): %s - retry %.1fs",
                                       attempt, self.max_retries, e, delay)
                        time.sleep(delay)
                        continue
                    logger.error("vLLM exception after %d retries: %s", self.max_retries, e)
                    return "", "error"

            if "error" in data:
                msg = data["error"].get("message", str(data["error"]))
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** (attempt - 1)) + random.random()
                    logger.warning("vLLM error (attempt %d/%d): %s - retry %.1fs",
                                   attempt, self.max_retries, msg, delay)
                    time.sleep(delay)
                    continue
                logger.error("vLLM error after %d retries: %s", self.max_retries, msg)
                return "", "error"

            if "choices" in data and data["choices"]:
                ch = data["choices"][0]
                text = ch.get("text", "")
                reason = ch.get("finish_reason", "stop")
                return text, reason

            logger.warning("Unrecognized vLLM response: %s",
                           json.dumps(data, ensure_ascii=False)[:200])
            return "", "error"

        logger.error("vLLM failed after %d retries", self.max_retries)
        return "", "error"


class VLLMChatClient:
    """Client for vLLM OpenAI-compatible Chat Completions API.
    Supports multiple base URLs with round-robin load balancing."""

    def __init__(self, api_urls, model="qwen-7b-instruct",
                 temperature=1.0, max_tokens=2048,
                 max_retries=5, retry_base_delay=1.0,
                 max_concurrent=16, api_key="EMPTY"):
        if isinstance(api_urls, str):
            api_urls = [u.strip() for u in api_urls.split(",")]
        self.base_urls = [u.rstrip("/") for u in api_urls]
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._local = threading.local()
        self._semaphore = threading.Semaphore(max_concurrent)
        self.api_key = api_key
        self._counter = 0
        self._counter_lock = threading.Lock()

    def _next_base_url(self):
        with self._counter_lock:
            url = self.base_urls[self._counter % len(self.base_urls)]
            self._counter += 1
            return url

    def _get_session(self):
        if not hasattr(self._local, 'session'):
            self._local.session = requests.Session()
            self._local.session.headers.update({
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            })
        return self._local.session

    def chat(self, messages, stop=None, temperature=None):
        temp = temperature if temperature is not None else self.temperature
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": temp,
            "top_p": 1.0,
        }
        if stop:
            payload["stop"] = stop

        session = self._get_session()

        for attempt in range(1, self.max_retries + 1):
            base = self._next_base_url()
            url = f"{base}/v1/chat/completions"
            with self._semaphore:
                try:
                    resp = session.post(url, json=payload, timeout=180)
                    data = resp.json()
                except Exception as e:
                    if attempt < self.max_retries:
                        delay = self.retry_base_delay * (2 ** (attempt - 1)) + random.random()
                        logger.warning("vLLM-chat exception (attempt %d/%d, %s): %s - retry %.1fs",
                                       attempt, self.max_retries, base, e, delay)
                        time.sleep(delay)
                        continue
                    logger.error("vLLM-chat exception after %d retries: %s", self.max_retries, e)
                    return "", "error"

            if "error" in data:
                msg = data["error"].get("message", str(data["error"]))
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** (attempt - 1)) + random.random()
                    logger.warning("vLLM-chat error (attempt %d/%d): %s - retry %.1fs",
                                   attempt, self.max_retries, msg, delay)
                    time.sleep(delay)
                    continue
                logger.error("vLLM-chat error after %d retries: %s", self.max_retries, msg)
                return "", "error"

            if "choices" in data and data["choices"]:
                ch = data["choices"][0]
                content = ch.get("message", {}).get("content", "")
                reason = ch.get("finish_reason", "stop")
                return content, reason

            logger.warning("Unrecognized vLLM-chat response: %s",
                           json.dumps(data, ensure_ascii=False)[:200])
            return "", "error"

        logger.error("vLLM-chat failed after %d retries", self.max_retries)
        return "", "error"

# ── Search Client ────────────────────────────────────────────────────────

MAX_DOC_CHARS = 800

class SearchClient:
    def __init__(self, search_url, topk=3, timeout=30):
        self.search_url, self.topk, self.timeout = search_url, topk, timeout
        self._local = threading.local()

    def _get_session(self):
        if not hasattr(self._local, 'session'):
            self._local.session = requests.Session()
        return self._local.session

    def search(self, query):
        if not query or not query.strip():
            return "No results found."
        try:
            r = self._get_session().post(self.search_url,
                                         json={"queries": [query], "topk": self.topk, "return_scores": True},
                                         timeout=self.timeout)
            r.raise_for_status()
            results = r.json().get("result", [[]])[0]
            return self._fmt(results)
        except Exception as e:
            logger.warning("Search failed for query '%s': %s", query[:60], e)
            return "No results found."

    @staticmethod
    def _fmt(results):
        parts = []
        for i, item in enumerate(results):
            doc = item.get("document", item)
            c = doc.get("contents", "")
            title = c.split("\n")[0] if c else ""
            text = "\n".join(c.split("\n")[1:]) if c else ""
            if len(text) > MAX_DOC_CHARS:
                text = text[:MAX_DOC_CHARS] + "..."
            parts.append(f"Doc {i+1}(Title: {title}) {text}")
        return "\n".join(parts) if parts else "No results found."

class DummySearchClient:
    def search(self, query): return "Search is not available."

# ── Multi-turn generation ────────────────────────────────────────────────

def extract_search_query(text):
    m = re.search(r"<search>(.*?)</search>", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"<search>(.*)", text, re.DOTALL)
    if m: return m.group(1).strip()
    return None

MAX_CONTEXT_CHARS = 12000

def _trim_context(messages, max_chars=MAX_CONTEXT_CHARS):
    """If total context is too long, keep the first user message and
    truncate middle turns, preserving the most recent turns."""
    total = sum(len(m["content"]) for m in messages)
    if total <= max_chars:
        return messages
    if len(messages) <= 2:
        return messages
    first = messages[:1]
    rest = messages[1:]
    while sum(len(m["content"]) for m in first + rest) > max_chars and len(rest) > 2:
        rest = rest[2:]
    return first + rest

CONTINUE_PROMPT = "Please continue your reasoning and provide a final answer in <answer></answer> tags."

def _ensure_alternating(messages):
    """Mediago API requires strict user/assistant alternation.
    If the last message is assistant, append a user nudge."""
    if messages and messages[-1]["role"] == "assistant":
        return messages + [{"role": "user", "content": CONTINUE_PROMPT}]
    return messages

def run_multiturn_generation(llm, searcher, prompt_text, max_turns=5, temperature=1.0):
    """Multi-turn generation for mediago (chat messages) backend."""
    messages = [{"role": "user", "content": prompt_text}]
    parts, turn_log = [], []

    for turn in range(max_turns):
        send_msgs = _trim_context(_ensure_alternating(messages))
        content, reason = llm.chat(send_msgs, stop=["</search>"], temperature=temperature)

        if not content and reason == "error":
            turn_log.append({"turn": turn+1, "action": "error"})
            break
        if not content:
            turn_log.append({"turn": turn+1, "action": "empty"})
            break

        has_open = "<search>" in content
        has_close = "</search>" in content

        if has_open and not has_close:
            content += "</search>"
            q = extract_search_query(content)
            sr = searcher.search(q) if q else "No results found."
            info = "<information>" + sr.strip() + "</information>"
            parts += [content, "\n\n" + info + "\n\n"]
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": info})
            turn_log.append({"turn": turn+1, "action": "search", "query": q})
            continue

        if has_open and has_close:
            q = extract_search_query(content)
            content = content[:content.index("</search>") + len("</search>")]
            sr = searcher.search(q) if q else "No results found."
            info = "<information>" + sr.strip() + "</information>"
            parts += [content, "\n\n" + info + "\n\n"]
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": info})
            turn_log.append({"turn": turn+1, "action": "search", "query": q})
            continue

        parts.append(content)
        messages.append({"role": "assistant", "content": content})
        if "<answer>" in content:
            if "</answer>" not in content:
                content += "</answer>"; parts[-1] = content
            turn_log.append({"turn": turn+1, "action": "answer"})
            break
        turn_log.append({"turn": turn+1, "action": "continue"})

    if not any("<answer>" in p for p in parts):
        send_msgs = _trim_context(_ensure_alternating(messages))
        final, _ = llm.chat(send_msgs, stop=None, temperature=temperature)
        if final:
            parts.append(final)
            turn_log.append({"turn": len(turn_log)+1, "action": "final_attempt"})

    return "".join(parts), turn_log


def run_multiturn_vllm(llm, searcher, prompt_text, max_turns=5, temperature=1.0):
    """Multi-turn generation for vLLM completions backend.
    Builds a single prompt string incrementally — matching RL training format."""
    accumulated = prompt_text
    parts, turn_log = [], []

    for turn in range(max_turns):
        content, reason = llm.complete(accumulated, stop=["</search>"],
                                       temperature=temperature)

        if not content and reason == "error":
            turn_log.append({"turn": turn+1, "action": "error"})
            break
        if not content:
            turn_log.append({"turn": turn+1, "action": "empty"})
            break

        has_open = "<search>" in content
        has_close = "</search>" in content

        if has_open and not has_close:
            content += "</search>"
            q = extract_search_query(content)
            sr = searcher.search(q) if q else "No results found."
            info = "<information>" + sr.strip() + "</information>"
            parts += [content, "\n\n" + info + "\n\n"]
            accumulated += content + "\n\n" + info + "\n\n"
            turn_log.append({"turn": turn+1, "action": "search", "query": q})
            continue

        if has_open and has_close:
            q = extract_search_query(content)
            content = content[:content.index("</search>") + len("</search>")]
            sr = searcher.search(q) if q else "No results found."
            info = "<information>" + sr.strip() + "</information>"
            parts += [content, "\n\n" + info + "\n\n"]
            accumulated += content + "\n\n" + info + "\n\n"
            turn_log.append({"turn": turn+1, "action": "search", "query": q})
            continue

        parts.append(content)
        accumulated += content
        if "<answer>" in content:
            if "</answer>" not in content:
                content += "</answer>"; parts[-1] = content
            turn_log.append({"turn": turn+1, "action": "answer"})
            break
        turn_log.append({"turn": turn+1, "action": "continue"})

    if not any("<answer>" in p for p in parts):
        final, _ = llm.complete(accumulated, stop=None, temperature=temperature)
        if final:
            parts.append(final)
            turn_log.append({"turn": len(turn_log)+1, "action": "final_attempt"})

    return "".join(parts), turn_log

# ── Per-question sampling ────────────────────────────────────────────────

_USE_VLLM = False

def sample_one_question(idx, qdata, llm, searcher, num_samples, max_turns, temperature):
    prompt_raw = qdata.get("prompt")
    if isinstance(prompt_raw, (list, np.ndarray)):
        prompt_text = (prompt_raw[0]["content"] if len(prompt_raw) > 0 and isinstance(prompt_raw[0], dict)
                       else str(prompt_raw[0]) if len(prompt_raw) > 0 else str(prompt_raw))
    else:
        prompt_text = str(prompt_raw)

    rm = qdata.get("reward_model", {})
    gt = rm.get("ground_truth", {}) if isinstance(rm, dict) else {}
    golden = gt.get("target", []) if isinstance(gt, dict) else []
    if hasattr(golden, "tolist"): golden = golden.tolist()
    elif isinstance(golden, str): golden = [golden]

    qid = qdata.get("id", f"q_{idx}")
    qtxt = qdata.get("question", "")
    dsrc = qdata.get("data_source", "unknown")

    results = []
    for si in range(num_samples):
        try:
            gen_fn = run_multiturn_vllm if _USE_VLLM else run_multiturn_generation
            resp, tlog = gen_fn(llm, searcher, prompt_text, max_turns, temperature)
            full = prompt_text + resp
            rew = compute_reward(full, golden)
            ans = extract_answer_from_response(resp)
            results.append({"question_id": qid, "question": qtxt, "data_source": dsrc,
                            "prompt": prompt_text, "response": resp, "answer": ans or "",
                            "reward": rew, "ground_truth": golden, "sample_idx": si,
                            "num_turns": len(tlog), "turn_log": tlog})
        except Exception as e:
            logger.error("Sample fail (q=%s s=%d): %s", qid, si, e)
            results.append({"question_id": qid, "question": qtxt, "data_source": dsrc,
                            "prompt": prompt_text, "response": "", "answer": "",
                            "reward": 0.0, "ground_truth": golden, "sample_idx": si,
                            "num_turns": 0, "turn_log": [], "error": str(e)})
    return results

# ── Data loading ─────────────────────────────────────────────────────────

def load_data(path, max_q=None):
    df = pd.read_parquet(path)
    logger.info("Loaded %s: %d rows", path, len(df))
    if max_q and max_q < len(df):
        df = df.head(max_q); logger.info("Truncated to %d", max_q)
    return df

def load_existing(path):
    done = set()
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try: done.add(json.loads(line)["question_id"])
                except: pass
        logger.info("Resume: %d questions done", len(done))
    return done

# ── Main ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Step 1: Rejection Sampling")
    p.add_argument("--data_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--max_questions", type=int, default=None)
    p.add_argument("--max_turns", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--api_url", default="http://osagw.simeji.me/gbu/rest/v1/ai_chat/openai_service")
    p.add_argument("--api_key", default="mediago_platform.oijio3f4893u2898")
    p.add_argument("--model", default="deploy_gpt5_chat")
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--search_url", default=None)
    p.add_argument("--search_topk", type=int, default=3)
    p.add_argument("--no_search", action="store_true")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    p.add_argument("--backend", default="mediago", choices=["mediago", "vllm", "vllm_chat"],
                   help="LLM backend: mediago (external API), vllm (local completions), vllm_chat (local chat API)")
    args = p.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    global _USE_VLLM
    if args.backend == "vllm":
        _USE_VLLM = True
        llm = VLLMClient(api_url=args.api_url, model=args.model,
                         temperature=args.temperature, max_tokens=args.max_tokens,
                         api_key=args.api_key, max_concurrent=args.workers * 2)
        logger.info("Backend: vLLM (%s model=%s)", args.api_url, args.model)
    elif args.backend == "vllm_chat":
        _USE_VLLM = False
        llm = VLLMChatClient(api_urls=args.api_url, model=args.model,
                             temperature=args.temperature, max_tokens=args.max_tokens,
                             api_key=args.api_key, max_concurrent=args.workers * 2)
        logger.info("Backend: vLLM-chat (%s model=%s)", args.api_url, args.model)
    else:
        headers = {
            "apikey": args.api_key,
            "User-Agent": "iAPI/1.0.0 (http://iapi.baidu-int.com)",
            "Accept": "*/*",
            "Host": "gbu.jp02-a30-apisix-sandbox.baidu-int.com",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
        }
        llm = LLMClient(api_url=args.api_url, headers=headers,
                         model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
        logger.info("Backend: mediago (%s model=%s)", args.api_url, args.model)

    logger.info("Testing API connectivity...")
    test_content, test_reason = llm.chat(
        [{"role": "user", "content": "Say hello in one word."}],
        temperature=0.1)
    if test_reason == "error" or not test_content:
        logger.error("API connectivity test FAILED. Check your --api_url, --api_key, --model.")
        logger.error("Response: content=%r reason=%r", test_content, test_reason)
        return
    logger.info("API OK. Test response: %s", test_content[:80])

    searcher = (DummySearchClient() if args.no_search or not args.search_url
                else SearchClient(args.search_url, args.search_topk))
    logger.info("Search: %s", "disabled" if isinstance(searcher, DummySearchClient) else args.search_url)

    df = load_data(args.data_file, args.max_questions)
    done = load_existing(args.output_file)
    pending = [(i, row.to_dict()) for i, row in df.iterrows() if row.get("id") not in done]
    logger.info("Pending: %d / %d", len(pending), len(df))
    if not pending:
        logger.info("Nothing to do"); return

    total_s, total_r, total_c = 0, 0.0, 0
    out_f = open(args.output_file, "a", encoding="utf-8")

    def process(item):
        return sample_one_question(item[0], item[1], llm, searcher,
                                   args.num_samples, args.max_turns, args.temperature)
    try:
        if args.workers <= 1:
            for item in tqdm(pending, desc="Sampling"):
                for r in process(item):
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    total_s += 1; total_r += r["reward"]; total_c += int(r["reward"] >= 1.0)
                out_f.flush()
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(process, it): it for it in pending}
                with tqdm(total=len(pending), desc="Sampling") as bar:
                    for f in as_completed(futs):
                        try:
                            for r in f.result():
                                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                                total_s += 1; total_r += r["reward"]; total_c += int(r["reward"] >= 1.0)
                            out_f.flush()
                        except Exception as e:
                            logger.error("Fail: %s", e)
                        bar.update(1)
    finally:
        out_f.close()

    logger.info("=" * 50)
    logger.info("Done. Samples=%d  Avg_reward=%.4f  Correct=%d (%.1f%%)",
                total_s, total_r / max(total_s, 1), total_c,
                100 * total_c / max(total_s, 1))
    logger.info("Output: %s", args.output_file)

if __name__ == "__main__":
    main()
