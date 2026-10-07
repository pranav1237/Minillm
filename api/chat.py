import json
import os
import sys
import tempfile
import traceback
import zipfile
from http.server import BaseHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ARTIFACT_ZIP = os.path.join(ROOT, "artifacts.zip")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "minillm-artifacts")
TOKENIZER_MODEL = os.path.join(CACHE_DIR, "tokenizer.model")
MODEL_FILE = os.path.join(CACHE_DIR, "tiny_llm.pt")
BLOCK_SIZE = 256

_MODEL = None
_TOKENIZER = None
_TORCH = None


def _load_torch():
    global _TORCH
    if _TORCH is None:
        import torch
        torch.set_num_threads(1)
        _TORCH = torch
    return _TORCH


def _ensure_artifacts():
    if os.path.exists(TOKENIZER_MODEL) and os.path.exists(MODEL_FILE):
        return
    if not os.path.exists(ARTIFACT_ZIP):
        raise RuntimeError("Trained artifacts.zip is not present in the deployed function bundle.")

    os.makedirs(CACHE_DIR, exist_ok=True)
    with zipfile.ZipFile(ARTIFACT_ZIP) as archive:
        names = archive.namelist()
        tokenizer_name = next((n for n in names if n.endswith("tokenizer.model")), None)
        model_name = next((n for n in names if n.endswith("tiny_llm.pt")), None)
        if not tokenizer_name or not model_name:
            raise RuntimeError("artifacts.zip does not contain tokenizer.model and tiny_llm.pt.")

        with archive.open(tokenizer_name) as src, open(TOKENIZER_MODEL, "wb") as dst:
            dst.write(src.read())
        with archive.open(model_name) as src, open(MODEL_FILE, "wb") as dst:
            dst.write(src.read())


def _get_runtime():
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    torch = _load_torch()
    _ensure_artifacts()

    from src.mini_llm.model import build_tiny_decoder_only_transformer
    from src.mini_llm.tokenizer_utils import load_tokenizer

    sp = load_tokenizer(TOKENIZER_MODEL)
    model = build_tiny_decoder_only_transformer(
        vocab_size=int(sp.get_piece_size()),
        max_len=BLOCK_SIZE,
    )
    state_dict = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise RuntimeError("The trained checkpoint is not a PyTorch state dictionary.")
    model.load_state_dict(state_dict)
    model.eval()
    _MODEL, _TOKENIZER = model, sp
    return _MODEL, _TOKENIZER


def _generate(prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
    torch = _load_torch()
    from src.mini_llm.tokenizer_utils import decode_ids, encode_text

    model, sp = _get_runtime()
    wrapped = f"<|user|> {prompt.strip()}\n<|assistant|>"
    token_ids = encode_text(sp, wrapped)

    # Preserve the model's 256-token context instead of allowing an oversized
    # prompt to make the Transformer fail before generation starts.
    token_ids = token_ids[-(BLOCK_SIZE - 1):]
    x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            x_cond = x[:, -BLOCK_SIZE:] if x.size(1) > BLOCK_SIZE else x
            logits = model(x_cond)[:, -1, :]
            logits = logits / max(temperature, 1e-5)

            if top_k > 0:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits[logits < values[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            x = torch.cat([x, next_id], dim=1)

    decoded = decode_ids(sp, x[0].tolist())
    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>", 1)[1]
    if "<|user|>" in decoded:
        decoded = decoded.split("<|user|>", 1)[0]
    return decoded.strip()


class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS, GET")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        # This probe is intentionally cheap. The first POST performs the actual
        # model load, so the UI can distinguish an available API route from a
        # loaded inference worker.
        self._send(
            200,
            {
                "ok": True,
                "service": "MiniLLM inference API",
                "model": "MiniLLM",
                "device": "cpu",
                "ready": True,
                "model_loaded": _MODEL is not None,
                "artifacts_present": os.path.exists(ARTIFACT_ZIP),
            },
        )

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return self._send(400, {"ok": False, "error": "Request body is empty."})
            if length > 20000:
                return self._send(413, {"ok": False, "error": "Prompt is too large."})

            raw = self.rfile.read(length)
            payload = json.loads(raw or b"{}")
            prompt = str(payload.get("prompt", "")).strip()
            if not prompt:
                return self._send(400, {"ok": False, "error": "Please enter a prompt."})

            max_new_tokens = min(max(int(payload.get("max_new_tokens", 80)), 1), 160)
            temperature = min(max(float(payload.get("temperature", 0.6)), 0.1), 2.0)
            top_k = min(max(int(payload.get("top_k", 30)), 0), 100)

            reply = _generate(prompt, max_new_tokens, temperature, top_k)
            if not reply:
                reply = "The trained model returned an empty completion. Try a shorter prompt."

            return self._send(
                200,
                {
                    "ok": True,
                    "reply": reply,
                    "model": "MiniLLM",
                    "tokens": max_new_tokens,
                },
            )
        except json.JSONDecodeError:
            return self._send(400, {"ok": False, "error": "Invalid JSON request."})
        except Exception as exc:
            # Always return JSON so the browser never fails with
            # “Unexpected token A … is not valid JSON”.
            message = str(exc).strip() or exc.__class__.__name__
            print("MiniLLM inference error:", traceback.format_exc())
            return self._send(
                500,
                {
                    "ok": False,
                    "error": message,
                    "type": exc.__class__.__name__,
                },
            )

    def log_message(self, *_args):
        return
