from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import math
import time
import json
import logging


# ----------------------------
# Utils
# ----------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    try:
        return float(x)
    except Exception:
        return default

def softmax(logits: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    t = max(1e-6, float(temperature))
    mx = max(logits.values()) if logits else 0.0
    exps = {k: math.exp((v - mx) / t) for k, v in logits.items()}
    s = sum(exps.values()) or 1.0
    return {k: v / s for k, v in exps.items()}

def entropy(probs: Dict[str, float]) -> float:
    e = 0.0
    for p in probs.values():
        if p > 0:
            e -= p * math.log(p + 1e-12)
    return e

def confidence_gap(probs: Dict[str, float]) -> float:
    if not probs:
        return 0.0
    vals = sorted(probs.values(), reverse=True)
    if len(vals) == 1:
        return vals[0]
    return vals[0] - vals[1]


# ----------------------------
# Core data structures
# ----------------------------

@dataclass(frozen=True)
class Hypothesis:
    name: str
    description: str = ""

@dataclass
class Rule:
    """
    A rule maps (signals, context) -> delta logits per hypothesis.

    Return format:
      {hypothesis_name: delta_logit, ...}
    """
    name: str
    apply: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, float]]
    strength: float = 1.0
    enabled: bool = True
    notes: str = ""

@dataclass
class QuantumConfig:
    temperature: float = 1.0
    min_confidence_gap: float = 0.10        # if top1-top2 < gap -> fallback to SAFE action
    max_entropy: float = 1.20               # if entropy too high -> uncertainty -> SAFE action
    safe_action: str = "WAIT"
    clamp_logit_abs: float = 12.0
    trace_limit: int = 200

@dataclass
class BeliefState:
    logits: Dict[str, float]
    probs: Dict[str, float]
    updated_at: float = field(default_factory=lambda: time.time())
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "updated_at": self.updated_at,
            "logits": dict(self.logits),
            "probs": dict(self.probs),
            "entropy": entropy(self.probs),
            "confidence_gap": confidence_gap(self.probs),
        }


# ----------------------------
# Quantum Module
# ----------------------------

class QuantumModule:
    """
    QCI Quantum Module Core V0.1
    - maintains a belief distribution over hypotheses
    - updates belief via rules (signal->delta logits)
    - chooses an action with safety gates (entropy/confidence)
    """

    def __init__(
        self,
        hypotheses: List[Hypothesis],
        rules: List[Rule],
        config: Optional[QuantumConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.hypotheses = {h.name: h for h in hypotheses}
        self.rules = rules
        self.cfg = config or QuantumConfig()
        self.log = logger or logging.getLogger("qci.quantum")

        # init logits = 0 for all hypotheses
        init_logits = {name: 0.0 for name in self.hypotheses.keys()}
        init_probs = softmax(init_logits, temperature=self.cfg.temperature)
        self.state = BeliefState(logits=init_logits, probs=init_probs)

        # ensure safe_action exists
        if self.cfg.safe_action not in self.hypotheses:
            self.hypotheses[self.cfg.safe_action] = Hypothesis(self.cfg.safe_action, "Default safe action")
            self.state.logits[self.cfg.safe_action] = 0.0
            self.state.probs = softmax(self.state.logits, temperature=self.cfg.temperature)

    def reset(self) -> None:
        for k in self.state.logits.keys():
            self.state.logits[k] = 0.0
        self.state.probs = softmax(self.state.logits, temperature=self.cfg.temperature)
        self.state.trace.clear()
        self.state.updated_at = time.time()

    def update(self, signals: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> BeliefState:
        ctx = context or {}
        trace_batch: List[Dict[str, Any]] = []

        # apply rules
        for rule in self.rules:
            if not rule.enabled:
                continue

            raw_delta = rule.apply(signals, ctx) or {}
            # scale by strength
            delta = {k: float(v) * float(rule.strength) for k, v in raw_delta.items()}

            # apply deltas to logits (only known hypotheses; unknown are ignored in v0.1)
            applied_any = False
            for h_name, d in delta.items():
                if h_name not in self.state.logits:
                    continue
                old = self.state.logits[h_name]
                new = old + d
                # clamp for numerical safety
                c = self.cfg.clamp_logit_abs
                new = max(-c, min(c, new))
                self.state.logits[h_name] = new
                applied_any = True

            if applied_any:
                trace_batch.append({
                    "type": "rule",
                    "rule": rule.name,
                    "strength": rule.strength,
                    "delta": delta,
                    "signals_hint": _compact_signals(signals),
                    "notes": rule.notes,
                    "ts": time.time(),
                })

        # normalize -> probs
        self.state.probs = softmax(self.state.logits, temperature=self.cfg.temperature)
        self.state.updated_at = time.time()

        # append trace with limit
        self.state.trace.extend(trace_batch)
        if len(self.state.trace) > self.cfg.trace_limit:
            self.state.trace = self.state.trace[-self.cfg.trace_limit:]

        return self.state

    def decide(self) -> Dict[str, Any]:
        probs = self.state.probs
        ent = entropy(probs)
        gap = confidence_gap(probs)

        # pick top hypothesis
        decision = max(probs.items(), key=lambda kv: kv[1])[0] if probs else self.cfg.safe_action
        safety_fallback = False
        reasons = []

        if ent > self.cfg.max_entropy:
            safety_fallback = True
            reasons.append(f"high_entropy={ent:.3f} > {self.cfg.max_entropy:.3f}")

        if gap < self.cfg.min_confidence_gap:
            safety_fallback = True
            reasons.append(f"low_confidence_gap={gap:.3f} < {self.cfg.min_confidence_gap:.3f}")

        if safety_fallback:
            decision = self.cfg.safe_action

        out = {
            "decision": decision,
            "beliefs": dict(probs),
            "health": {
                "entropy": ent,
                "confidence_gap": gap,
                "safety_fallback": safety_fallback,
                "reasons": reasons,
            },
            "trace_tail": self.state.trace[-10:],  # last 10 events
            "state": self.state.snapshot(),
        }
        return out

    def explain_json(self, last_n: int = 30) -> str:
        payload = {
            "state": self.state.snapshot(),
            "trace": self.state.trace[-max(1, last_n):],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


def _compact_signals(signals: Dict[str, Any], max_items: int = 12) -> Dict[str, Any]:
    # keep only a few keys to avoid huge trace
    items = list(signals.items())[:max_items]
    compact = {}
    for k, v in items:
        if isinstance(v, (int, float, bool, str)) and len(str(v)) < 80:
            compact[k] = v
        else:
            compact[k] = str(type(v))
    return compact


# ----------------------------
# Example rules (starter pack)
# ----------------------------

def rule_price_prediction() -> Rule:
    """
    Expects signals:
      - pred_net_usd: expected net profit (or edge) in USD
    """
    def apply(signals: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, float]:
        edge = _safe_float(signals.get("pred_net_usd", 0.0))
        # simple shape: push BUY if edge positive, SELL if negative
        return {
            "BUY": max(0.0, edge) * 0.8,
            "SELL": max(0.0, -edge) * 0.8,
        }
    return Rule(name="price_prediction", apply=apply, strength=1.0, notes="Uses pred_net_usd as directional edge")

def rule_whale_activity() -> Rule:
    """
    Expects signals:
      - whale_score: [-1..+1] (negative bearish, positive bullish)
    """
    def apply(signals: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, float]:
        w = _safe_float(signals.get("whale_score", 0.0))
        return {
            "BUY": max(0.0, w) * 0.6,
            "SELL": max(0.0, -w) * 0.6,
        }
    return Rule(name="whale_activity", apply=apply, strength=1.0, notes="Whale sentiment pushes BUY/SELL")

def rule_risk_off() -> Rule:
    """
    Expects signals:
      - volatility: [0..inf] (higher -> more risk)
      - llm_risk_flag: bool (LLM says 'danger'/news risk)
    """
    def apply(signals: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, float]:
        vol = _safe_float(signals.get("volatility", 0.0))
        llm_flag = _safe_float(signals.get("llm_risk_flag", False))
        risk = min(2.0, vol) * 0.7 + llm_flag * 1.2
        return {
            "RISK_OFF": risk,
            "WAIT": risk * 0.5,
            "BUY": -risk * 0.4,
            "SELL": -risk * 0.2,  # sometimes sell is fine, but in v0.1 we simply suppress actions
        }
    return Rule(name="risk_off", apply=apply, strength=1.0, notes="Volatility/LLM-risk push module to safety")

def build_default_quantum_v01() -> QuantumModule:
    hyps = [
        Hypothesis("BUY", "Enter long / buy"),
        Hypothesis("SELL", "Enter short / sell / exit"),
        Hypothesis("WAIT", "Do nothing"),
        Hypothesis("RISK_OFF", "Reduce exposure / stop trading"),
    ]
    rules = [rule_price_prediction(), rule_whale_activity(), rule_risk_off()]
    cfg = QuantumConfig(
        temperature=1.0,
        min_confidence_gap=0.12,
        max_entropy=1.25,
        safe_action="WAIT",
    )
    return QuantumModule(hypotheses=hyps, rules=rules, config=cfg)
