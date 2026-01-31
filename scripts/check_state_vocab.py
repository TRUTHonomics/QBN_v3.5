#!/usr/bin/env python3
"""
State Vocabulary Health Check voor QBN_v3.5.

Doel:
- Detecteer inconsistenties in state-namen (casing/typos) die kunnen leiden tot
  missing states (bv. bearish nooit geraakt) of CPT mismatch.
- Controleer dat de network-structure node states overeenkomen met de canonical sets.

Scope (bewust beperkt):
- AST-scan van Python source files op string literals die op states lijken.
- Validatie van `inference/network_structure.py` (NodeDefinition.state lists).

Logging:
- Volgt project logging patroon via `core.logging_utils.setup_logging`.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import sys
import importlib.util
import types

# REASON: Scripts draaien vanuit /scripts; voeg project root toe aan sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_utils import setup_logging

logger = setup_logging("check_state_vocab")


DEFAULT_SCAN_DIRS = [
    "analysis",
    "config",
    "core",
    "inference",
    "menus",
    "scripts",
    "validation",
]


@dataclass(frozen=True)
class Finding:
    file_path: Path
    line: int
    value: str
    message: str


def _get_canonical_state_sets() -> Tuple[Set[str], Set[str]]:
    """
    Returns:
        (canonical_states, legacy_aliases)
    """
    # Canonical sets are derived from the network structure and enums.
    # REASON: Single-source-of-truth must be the implemented DAG and node_types.
    # IMPORTANT: We load modules by file path to avoid importing `inference/__init__.py`
    # (which pulls heavy deps like pandas) for this lightweight lint-like script.

    def ensure_pkg(pkg_name: str, pkg_path: Path):
        # REASON: Maak een minimal package aan zodat relative imports werken,
        # zonder inference/__init__.py uit te voeren (heavy deps).
        if pkg_name in sys.modules:
            return
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(pkg_path)]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

    def load_module(mod_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module {mod_name} from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module

    ensure_pkg("inference", PROJECT_ROOT / "inference")
    node_types_mod = load_module("inference.node_types", PROJECT_ROOT / "inference" / "node_types.py")

    CompositeState = node_types_mod.CompositeState
    OutcomeState = node_types_mod.OutcomeState
    BarrierOutcomeState = node_types_mod.BarrierOutcomeState
    RegimeState = node_types_mod.RegimeState

    canonical_states: Set[str] = set()

    # Network structure (may require networkx; if unavailable we still validate enums)
    try:
        network_structure_mod = load_module("inference.network_structure", PROJECT_ROOT / "inference" / "network_structure.py")
        structure = network_structure_mod.QBNv3NetworkStructure()
        for node in structure.nodes.values():
            canonical_states.update(node.states)
    except Exception as e:
        logger.warning(f"Kon network structure niet laden (fallback naar enums): {e}")

    # Also include enum state values/names explicitly (defensive).
    canonical_states.update({s.value for s in CompositeState})
    canonical_states.update(set(OutcomeState.state_names()))
    canonical_states.update({s.value for s in BarrierOutcomeState})
    canonical_states.update(set(RegimeState.all_states()))

    # Legacy aliases that we intentionally accept as inputs in some mappings.
    # EXPL: Some components accept capitalized strings originating from older SignalState
    # conventions. These should not be used as BN node state definitions.
    legacy_aliases = {
        "Strong_Bullish",
        "Bullish",
        "Neutral",
        "Bearish",
        "Strong_Bearish",
        # Lowercase outcome aliases used for case-insensitive matching in validators
        "slight_bullish",
        "slight_bearish",
    }

    return canonical_states, legacy_aliases


def _is_probable_state_literal(s: str) -> bool:
    """
    Heuristic filter to avoid scanning irrelevant literals.
    """
    if not s or len(s) > 64:
        return False
    if " " in s:
        return False
    # REASON: Alle-uppercase tokens worden vaak gebruikt als labels/categorieën,
    # niet als concrete state values.
    if s.isupper():
        return False
    s_lower = s.lower()

    # REASON: Sluit config/kolom-namen uit die vaak bullish/bearish tokens bevatten,
    # maar geen states zijn (voorkomt spam).
    if any(k in s_lower for k in ("threshold", "neutral_band", "time_to_", "_atr", "max_up", "max_down", "p_up_", "p_down_")):
        return False
    if s_lower.startswith(("time_to_", "max_", "p_")):
        return False
    # Common state-like tokens in this repo.
    tokens = (
        "bullish",
        "bearish",
        "neutral",
        "strong_",
        "slight_",
        "sync_",
        "macro_",
        "exit_",
        "low_vol",
        "high_vol",
        "target_hit",
        "stoploss_hit",
        "timeout",
        "weak_",
        "strong_",
        "no_setup",
        "up_",
        "down_",
    )
    return any(t in s_lower for t in tokens)


class _LiteralCollector(ast.NodeVisitor):
    def __init__(self):
        self.literals: List[Tuple[int, str]] = []

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            self.literals.append((getattr(node, "lineno", 0) or 0, node.value))
        self.generic_visit(node)


def _iter_python_files(scan_dirs: Iterable[str]) -> Iterable[Path]:
    for rel in scan_dirs:
        p = PROJECT_ROOT / rel
        if not p.exists():
            continue
        for file_path in p.rglob("*.py"):
            # Skip caches and hidden dirs
            if "__pycache__" in file_path.parts:
                continue
            if any(part.startswith(".") for part in file_path.parts):
                continue
            yield file_path


def _scan_file_for_state_literals(file_path: Path) -> List[Tuple[int, str]]:
    src = file_path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src, filename=str(file_path))

    # Remove docstrings from consideration to reduce false positives.
    def strip_docstring(body: List[ast.stmt]) -> List[ast.stmt]:
        if not body:
            return body
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant):
            if isinstance(first.value.value, str):
                return body[1:]
        return body

    tree.body = strip_docstring(tree.body)
    for node in ast.walk(tree):
        # Strip function/class docstrings by mutating their bodies
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.body = strip_docstring(node.body)

    collector = _LiteralCollector()
    collector.visit(tree)
    return collector.literals


def _validate_network_structure_states() -> List[Finding]:
    # REASON: Load by file path to avoid importing inference/__init__.py (heavy deps),
    # while still supporting relative imports inside inference modules.
    def ensure_pkg(pkg_name: str, pkg_path: Path):
        if pkg_name in sys.modules:
            return
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(pkg_path)]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

    def load_module(mod_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"spec missing for {mod_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module

    ensure_pkg("inference", PROJECT_ROOT / "inference")
    node_types_mod = load_module("inference.node_types", PROJECT_ROOT / "inference" / "node_types.py")
    CompositeState = node_types_mod.CompositeState
    OutcomeState = node_types_mod.OutcomeState

    try:
        network_structure_mod = load_module("inference.network_structure", PROJECT_ROOT / "inference" / "network_structure.py")
        QBNv3NetworkStructure = network_structure_mod.QBNv3NetworkStructure
    except Exception as e:
        return [Finding(file_path=PROJECT_ROOT / "inference" / "network_structure.py", line=0, value="", message=f"Kon network_structure niet laden: {e}")]

    findings: List[Finding] = []
    structure = QBNv3NetworkStructure()

    composite_states = {s.value for s in CompositeState}
    outcome_states = set(OutcomeState.state_names())

    for node_name, node in structure.nodes.items():
        # Basic integrity
        if not node.states:
            findings.append(
                Finding(
                    file_path=PROJECT_ROOT / "inference" / "network_structure.py",
                    line=0,
                    value=node_name,
                    message="Node heeft lege states lijst (CPT mismatch risico).",
                )
            )
            continue

        # Composite nodes must use lowercase composite states exactly.
        if node_name.endswith("_Composite"):
            bad = [s for s in node.states if s not in composite_states]
            if bad:
                findings.append(
                    Finding(
                        file_path=PROJECT_ROOT / "inference" / "network_structure.py",
                        line=0,
                        value=",".join(bad),
                        message=f"{node_name} bevat niet-canonical composite state(s). Verwacht: {sorted(composite_states)}",
                    )
                )

        # Prediction_* nodes must use OutcomeState.state_names() (TitleCase)
        if node_name.startswith("Prediction_"):
            bad = [s for s in node.states if s not in outcome_states]
            if bad:
                findings.append(
                    Finding(
                        file_path=PROJECT_ROOT / "inference" / "network_structure.py",
                        line=0,
                        value=",".join(bad),
                        message=f"{node_name} bevat niet-canonical outcome state(s). Verwacht: {sorted(outcome_states)}",
                    )
                )

    return findings


def run_check(scan_dirs: Optional[List[str]] = None) -> List[Finding]:
    scan_dirs = scan_dirs or DEFAULT_SCAN_DIRS

    canonical, legacy_aliases = _get_canonical_state_sets()
    canonical_lower = {s.lower(): s for s in canonical}

    findings: List[Finding] = []
    findings.extend(_validate_network_structure_states())

    for file_path in _iter_python_files(scan_dirs):
        try:
            literals = _scan_file_for_state_literals(file_path)
        except SyntaxError as e:
            findings.append(
                Finding(
                    file_path=file_path,
                    line=getattr(e, "lineno", 0) or 0,
                    value="",
                    message=f"SyntaxError tijdens AST parse: {e}",
                )
            )
            continue
        except Exception as e:
            findings.append(
                Finding(
                    file_path=file_path,
                    line=0,
                    value="",
                    message=f"Onverwachte fout tijdens scan: {e}",
                )
            )
            continue

        for lineno, value in literals:
            if not _is_probable_state_literal(value):
                continue

            if value in canonical:
                continue
            if value in legacy_aliases:
                continue

            # Case/format mismatch: same lower but not exact
            val_lower = value.lower()
            if val_lower in canonical_lower:
                findings.append(
                    Finding(
                        file_path=file_path,
                        line=lineno,
                        value=value,
                        message=f"State literal casing/format mismatch. Bedoelde canonical: '{canonical_lower[val_lower]}'",
                    )
                )
                continue

    return findings


def main():
    parser = argparse.ArgumentParser(description="QBN state vocabulary check (v3.5)")
    parser.add_argument(
        "--scan-dirs",
        type=str,
        default=",".join(DEFAULT_SCAN_DIRS),
        help="Comma-separated lijst van directories (relatief aan project root) om te scannen.",
    )
    parser.add_argument(
        "--fail-on-issue",
        action="store_true",
        help="Exit code 1 als er findings zijn.",
    )
    args = parser.parse_args()

    scan_dirs = [d.strip() for d in args.scan_dirs.split(",") if d.strip()]
    findings = run_check(scan_dirs=scan_dirs)

    if not findings:
        logger.info("Geen state-vocab issues gevonden.")
        return 0

    logger.warning(f"Gevonden issues: {len(findings)}")
    for f in findings[:200]:
        loc = f"{f.file_path.relative_to(PROJECT_ROOT)}:{f.line}" if f.line else str(f.file_path.relative_to(PROJECT_ROOT))
        logger.warning(f"{loc}: '{f.value}' -> {f.message}")

    if len(findings) > 200:
        logger.warning(f"... truncated, total findings={len(findings)}")

    return 1 if args.fail_on_issue else 0


if __name__ == "__main__":
    raise SystemExit(main())

