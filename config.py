"""Central configuration for VehicleGPT.

Every value can be overridden with a ``VGPT_<NAME>`` environment variable
(see ``.env.example``). Nothing in this module touches the network.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = "1.0.0"

# Clearly fictitious defaults so the demo runs out of the box.
# app.py logs a loud warning while any of them is still in use.
DEV_DRIVER_TOKEN = "dev-driver-token-CHANGE-ME"
DEV_SECRET = "dev-secret-CHANGE-ME"
DEV_SERVICE_PIN = "000000"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"VGPT_{name}", default)


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(f"VGPT_{name}")
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # ---- paths -----------------------------------------------------------
    db_path: str = str(ROOT / "data" / "vehicle.db")
    index_dir: str = str(ROOT / "data" / "index")
    knowledge_dir: str = str(ROOT / "knowledge")
    audit_path: str = str(ROOT / "data" / "audit.jsonl")
    feedback_path: str = str(ROOT / "data" / "intent_feedback.jsonl")
    model_cache: str = str(ROOT / "models")

    # ---- locale ----------------------------------------------------------
    tz: str = "Asia/Kolkata"

    # ---- MQTT (internal bus, localhost only) ------------------------------
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_user: str = "vgpt"
    mqtt_password: str = "fictitious-mqtt-password"

    # ---- LLM (Ollama) ----------------------------------------------------
    llm_enabled: bool = True
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "llama3.2:3b"          # use llama3.2:1b-instruct-q4_K_M for the 2 GB profile
    llm_num_ctx: int = 4096
    llm_temperature: float = 0.1
    llm_num_predict: int = 200               # hard cap on generated tokens (bounds latency)
    llm_timeout_s: float = 45.0

    # ---- retrieval / routing ---------------------------------------------
    embed_backend: str = "auto"              # auto | fastembed | hashing
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_enabled: bool = False
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    router_threshold: float = -1.0           # <0 means "use the backend default"
    router_floor: float = -1.0               # <0 means "use the backend default"
    evidence_token_budget: int = 1000

    # ---- storage ---------------------------------------------------------
    hot_days: int = 7
    warm_days: int = 70                      # 10 weeks > the 8-week requirement
    disk_guard_pct: float = 85.0
    idle_kph: float = 1.0                    # minute-mean speed below this = idle
    highway_kph: float = 60.0                # minute-mean speed at/above this = highway

    # ---- API / security ---------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    driver_token: str = DEV_DRIVER_TOKEN
    secret: str = DEV_SECRET
    service_pin: str = DEV_SERVICE_PIN
    service_token_ttl_s: int = 900
    query_max_chars: int = 400

    def insecure_defaults(self) -> list[str]:
        bad = []
        if self.driver_token == DEV_DRIVER_TOKEN:
            bad.append("VGPT_DRIVER_TOKEN")
        if self.secret == DEV_SECRET:
            bad.append("VGPT_SECRET")
        if self.service_pin == DEV_SERVICE_PIN:
            bad.append("VGPT_SERVICE_PIN")
        return bad

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            db_path=_env("DB_PATH", d.db_path),
            index_dir=_env("INDEX_DIR", d.index_dir),
            knowledge_dir=_env("KNOWLEDGE_DIR", d.knowledge_dir),
            audit_path=_env("AUDIT_PATH", d.audit_path),
            feedback_path=_env("FEEDBACK_PATH", d.feedback_path),
            model_cache=_env("MODEL_CACHE", d.model_cache),
            tz=_env("TZ", d.tz),
            mqtt_host=_env("MQTT_HOST", d.mqtt_host),
            mqtt_port=int(_env("MQTT_PORT", str(d.mqtt_port))),
            mqtt_user=_env("MQTT_USER", d.mqtt_user),
            mqtt_password=_env("MQTT_PASSWORD", d.mqtt_password),
            llm_enabled=_bool("LLM_ENABLED", d.llm_enabled),
            ollama_url=_env("OLLAMA_URL", d.ollama_url),
            llm_model=_env("LLM_MODEL", d.llm_model),
            llm_num_ctx=int(_env("LLM_NUM_CTX", str(d.llm_num_ctx))),
            llm_temperature=float(_env("LLM_TEMPERATURE", str(d.llm_temperature))),
            llm_num_predict=int(_env("LLM_NUM_PREDICT", str(d.llm_num_predict))),
            llm_timeout_s=float(_env("LLM_TIMEOUT_S", str(d.llm_timeout_s))),
            embed_backend=_env("EMBED_BACKEND", d.embed_backend),
            embed_model=_env("EMBED_MODEL", d.embed_model),
            rerank_enabled=_bool("RERANK_ENABLED", d.rerank_enabled),
            rerank_model=_env("RERANK_MODEL", d.rerank_model),
            router_threshold=float(_env("ROUTER_THRESHOLD", str(d.router_threshold))),
            router_floor=float(_env("ROUTER_FLOOR", str(d.router_floor))),
            evidence_token_budget=int(_env("EVIDENCE_TOKEN_BUDGET", str(d.evidence_token_budget))),
            hot_days=int(_env("HOT_DAYS", str(d.hot_days))),
            warm_days=int(_env("WARM_DAYS", str(d.warm_days))),
            disk_guard_pct=float(_env("DISK_GUARD_PCT", str(d.disk_guard_pct))),
            idle_kph=float(_env("IDLE_KPH", str(d.idle_kph))),
            highway_kph=float(_env("HIGHWAY_KPH", str(d.highway_kph))),
            api_host=_env("API_HOST", d.api_host),
            api_port=int(_env("API_PORT", str(d.api_port))),
            driver_token=_env("DRIVER_TOKEN", d.driver_token),
            secret=_env("SECRET", d.secret),
            service_pin=_env("SERVICE_PIN", d.service_pin),
            service_token_ttl_s=int(_env("SERVICE_TOKEN_TTL_S", str(d.service_token_ttl_s))),
            query_max_chars=int(_env("QUERY_MAX_CHARS", str(d.query_max_chars))),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
