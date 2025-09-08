from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd

try:  # Prefer relative imports when available
    from ..config import MASTER_TOKEN_LIST
    from ..core.dataframe_registry import DataFrameRegistry
    from ..database.sql_engine import RawDataSQLEngine
except Exception:  # pragma: no cover - fallback for direct execution context
    from unicorn_wealth.config import MASTER_TOKEN_LIST  # type: ignore
    from unicorn_wealth.core.dataframe_registry import DataFrameRegistry  # type: ignore
    from unicorn_wealth.database.sql_engine import RawDataSQLEngine  # type: ignore

__all__ = ["UnifiedFeatureEngine"]


@dataclass(frozen=True)
class FeatureDef:
    operation: str
    step_order: int
    calculate_per_token: bool
    transform_data_source: str
    df_variable_id: Optional[str]
    df_frame_store: bool
    df_update_mode: Optional[str]
    df_storage_period: Optional[int]
    live_cadence: Optional[str]


def _safe_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    return s in {"true", "t", "1", "yes", "y"}


def _safe_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        i = int(str(val).strip())
        return i
    except Exception:
        return None


def _parse_possible_sources(text: str) -> List[str]:
    """Heuristically parse potential DataFrame source names from text.

    We consider tokens that either contain the placeholder "{t}" or end with
    "_df" as valid source identifiers. Multiple sources are frequently
    space-delimited in the spec.
    """
    if not text:
        return []
    rough_tokens = [tok.strip() for tok in text.replace("\n", " ").split(" ")]
    sources: List[str] = []
    for tok in rough_tokens:
        if not tok:
            continue
        # Strip simple punctuation wrappers
        tok = tok.strip(",;()[]{}")
        if "{t}" in tok or tok.endswith("_df"):
            sources.append(tok)
    return sources


class _SimpleDiGraph:
    """Minimal directed graph with Kahn topological sort and attributes."""

    def __init__(self) -> None:
        self._adj: Dict[str, set[str]] = {}
        self._in_deg: Dict[str, int] = {}
        self._attrs: Dict[str, Dict[str, Any]] = {}

    def add_node(self, node: str, **attrs: Any) -> None:
        if node not in self._adj:
            self._adj[node] = set()
            self._in_deg[node] = 0
            self._attrs[node] = {}
        self._attrs[node].update(attrs)

    def add_edge(self, u: str, v: str) -> None:
        if u not in self._adj:
            self.add_node(u)
        if v not in self._adj:
            self.add_node(v)
        if v not in self._adj[u]:
            self._adj[u].add(v)
            self._in_deg[v] += 1

    def nodes(self) -> Iterable[str]:
        return self._adj.keys()

    def attrs(self, node: str) -> Dict[str, Any]:
        return self._attrs.get(node, {})

    def topological_sort(self, stable_key: Optional[Any] = None) -> List[str]:
        # Kahn's algorithm with stable ordering by provided key
        in_deg = dict(self._in_deg)
        zero = [n for n, d in in_deg.items() if d == 0]

        def sort_key(n: str) -> Tuple[Any, str]:
            if stable_key is None:
                return (0, n)
            # step_order primary if available
            step = self._attrs.get(n, {}).get("step_order")
            return (step if step is not None else 0, n)

        zero.sort(key=sort_key)
        order: List[str] = []
        while zero:
            n = zero.pop(0)
            order.append(n)
            for m in sorted(self._adj[n]):
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    zero.append(m)
                    zero.sort(key=sort_key)
        if len(order) != len(in_deg):
            # Cycle detected; fall back to insertion order of nodes as last resort
            remaining = [n for n in self._adj.keys() if n not in order]
            order.extend(remaining)
        return order


def _replace_token_placeholder(name: str, token: str) -> str:
    return name.replace("{t}", token)


def _ensure_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    if "timestamp" in dfc.columns:
        return dfc
    if dfc.index.name == "timestamp":
        dfc = dfc.reset_index()
    else:
        # Try common index types; if not, create a simple range timestamp
        dfc = dfc.reset_index()
        if "timestamp" not in dfc.columns:
            dfc["timestamp"] = np.arange(len(dfc), dtype=int)
    return dfc


def _merge_on_timestamp(dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame(columns=["timestamp"]).astype(
            {"timestamp": "datetime64[ns]"}
        )
    working = _ensure_timestamp_index(dfs[0])
    for d in dfs[1:]:
        working = working.merge(
            _ensure_timestamp_index(d),
            on="timestamp",
            how="outer",
            suffixes=("", "_r"),
        )
    # Sort by timestamp if possible
    if pd.api.types.is_datetime64_any_dtype(
        working["timestamp"]
    ) or pd.api.types.is_numeric_dtype(working["timestamp"]):
        working = working.sort_values("timestamp")
    # Forward fill then zero-fill for numeric columns
    working = working.ffill()
    num_cols = working.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        working[num_cols] = working[num_cols].fillna(0)
    # Remaining NaNs (non-numeric) -> keep as is to avoid unintended casts
    return working


def _worker_compute_for_token(
    token: str,
    ordered_ops: List[str],
    feature_map: Dict[str, FeatureDef],
    base_inputs: Dict[str, pd.DataFrame],
) -> List[Tuple[str, pd.DataFrame, Optional[str], Optional[int]]]:
    """Worker process: compute placeholder features for a single token.

    Returns a list of tuples (store_name, df, update_mode, storage_period)
    for features that should be stored.
    """
    results: List[Tuple[str, pd.DataFrame, Optional[str], Optional[int]]] = []

    produced: Dict[str, pd.DataFrame] = {}
    produced.update(base_inputs)

    for op in ordered_ops:
        f = feature_map.get(op)
        if not f:
            continue
        if not f.calculate_per_token:
            continue

        # Collect inputs inferred from transform_data_source
        src_names = _parse_possible_sources(f.transform_data_source)
        token_src_names = [_replace_token_placeholder(s, token) for s in src_names]
        input_dfs: List[pd.DataFrame] = []
        for name in token_src_names:
            df = produced.get(name)
            if df is not None:
                input_dfs.append(df)
        merged = _merge_on_timestamp(input_dfs) if input_dfs else pd.DataFrame()

        # Placeholder calculation: create a simple series defaulting to 0
        if merged.empty:
            # fabricate minimal frame
            merged = pd.DataFrame({"timestamp": pd.to_datetime([])})
        out = merged.copy()
        col_name = f.operation
        if col_name in out.columns:
            # Do not overwrite if already present
            pass
        else:
            # Create a placeholder numeric column
            out[col_name] = 0.0

        # Keep the produced df available under a conventional name so later
        # steps can reference it. Use df_variable_id if it looks like a df
        # name; else use operation as key
        produced_key = f.df_variable_id or f"{token}_{f.operation}_df"
        produced[produced_key] = out

        # Queue storage if required
        if f.df_frame_store:
            store_name = produced_key
            results.append((store_name, out, f.df_update_mode, f.df_storage_period))

    return results


class UnifiedFeatureEngine:
    """Core orchestrator for feature engineering using a DAG of operations.

    This engine reads the Unicorn_Wealth_Feature_Set.json specification,
    constructs a DAG of feature operations, computes a topological execution
    order, and orchestrates historical processing per token in parallel.

    Note: feature computation itself is placeholder logic; this module focuses
    on orchestration, dependency handling, and I/O coordination.
    """

    def __init__(
        self,
        registry: DataFrameRegistry,
        sql_engine: Optional[RawDataSQLEngine] = None,
        feature_spec_path: Optional[str] = None,
    ) -> None:
        self._registry = registry
        self._sql_engine = sql_engine
        self._feature_spec_path = feature_spec_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "specifications",
            "Unicorn_Wealth_Feature_Set.json",
        )

        self._features: List[FeatureDef] = self._load_spec(self._feature_spec_path)
        self._feature_map: Dict[str, FeatureDef] = {
            f.operation: f for f in self._features
        }
        self._dag = self._build_dag(self._features)
        self._ordered_ops: List[str] = self._topological_sort_features()

    @staticmethod
    def _load_spec(path: str) -> List[FeatureDef]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        feats: List[FeatureDef] = []
        for item in raw:
            feats.append(
                FeatureDef(
                    operation=str(item.get("operation")),
                    step_order=int(item.get("step_order", 0)),
                    calculate_per_token=bool(item.get("calculate_per_token", True)),
                    transform_data_source=str(item.get("transform_data_source", "")),
                    df_variable_id=(
                        None
                        if str(item.get("df_variable_id", "")).strip().lower()
                        in {"", "not required"}
                        else str(item.get("df_variable_id"))
                    ),
                    df_frame_store=_safe_bool(item.get("df_frame_store")),
                    df_update_mode=(
                        None
                        if str(item.get("df_update_mode", "")).strip().lower()
                        in {"", "not required"}
                        else str(item.get("df_update_mode"))
                    ),
                    df_storage_period=_safe_int(item.get("df_storage_period")),
                    live_cadence=(
                        None
                        if str(item.get("live_cadence", "")).strip().lower()
                        in {"", "not required"}
                        else str(item.get("live_cadence"))
                    ),
                )
            )
        return feats

    def _build_dag(self, features: Sequence[FeatureDef]) -> _SimpleDiGraph:
        # Try to use networkx if available for future extensibility;
        # fallback otherwise
        try:  # pragma: no cover - optional dependency
            import networkx as nx  # type: ignore

            g = nx.DiGraph()
            for f in features:
                g.add_node(f.operation, step_order=f.step_order)
            for f in features:
                srcs = _parse_possible_sources(f.transform_data_source)
                for s in srcs:
                    g.add_edge(s, f.operation)
            # Wrap networkx graph with a thin adapter for topological sort compatibility
            dg = _SimpleDiGraph()
            for n, attrs in g.nodes(data=True):
                dg.add_node(str(n), **attrs)
            for u, v in g.edges():
                dg.add_edge(str(u), str(v))
            return dg
        except Exception:
            pass

        dg = _SimpleDiGraph()
        for f in features:
            dg.add_node(f.operation, step_order=f.step_order)
        for f in features:
            srcs = _parse_possible_sources(f.transform_data_source)
            for s in srcs:
                dg.add_node(s)
                dg.add_edge(s, f.operation)
        return dg

    def _topological_sort_features(self) -> List[str]:
        order = self._dag.topological_sort(stable_key=True)
        # We only execute known feature operations (exclude raw source nodes)
        ops_set = set(self._feature_map.keys())
        ordered_ops = [n for n in order if n in ops_set]
        # Stable secondary sort by step_order for nodes on the same level
        ordered_ops.sort(key=lambda op: (self._feature_map[op].step_order, op))
        return ordered_ops

    async def _gather_base_inputs_for_token(
        self, token: str
    ) -> Dict[str, pd.DataFrame]:
        """Fetch base inputs from the registry for a token.

        Inputs are inferred from transform_data_source in the spec.
        """
        names: set[str] = set()
        for f in self._features:
            srcs = _parse_possible_sources(f.transform_data_source)
            for s in srcs:
                concrete = _replace_token_placeholder(s, token)
                names.add(concrete)
        inputs: Dict[str, pd.DataFrame] = {}
        for name in names:
            try:
                inputs[name] = await self._registry.get_df(name)
            except KeyError:
                # Input might be a descriptive string or not yet produced; skip
                continue
        return inputs

    async def run_historical(self, cadence: Optional[str] = None) -> None:
        """Run historical computation for all tokens in parallel.

        This will:
        - Determine ordered features via topological sort.
        - For each token, prefetch base inputs from the registry.
        - Use a ProcessPoolExecutor where each process computes placeholder
          features for a single token.
        - Collect results and store DataFrames via the registry when required.
        """
        # Filter operations by cadence if provided
        if cadence is not None:
            filtered_ops = [
                f.operation
                for f in self._features
                if f.operation in self._ordered_ops
                and (f.live_cadence is None or f.live_cadence == cadence)
            ]
        else:
            filtered_ops = list(self._ordered_ops)

        feature_map = self._feature_map
        tokens = list(MASTER_TOKEN_LIST.keys())

        # Prefetch inputs per token using the registry (async)
        base_inputs_per_token = await asyncio.gather(
            *[self._gather_base_inputs_for_token(t) for t in tokens]
        )
        token_to_inputs = {t: base_inputs_per_token[i] for i, t in enumerate(tokens)}

        loop = asyncio.get_running_loop()
        results_per_token: List[
            Tuple[str, List[Tuple[str, pd.DataFrame, Optional[str], Optional[int]]]]
        ] = []

        def submit_one(
            tok: str,
        ) -> Tuple[str, List[Tuple[str, pd.DataFrame, Optional[str], Optional[int]]]]:
            outs = _worker_compute_for_token(
                tok,
                filtered_ops,
                feature_map,
                token_to_inputs.get(tok, {}),
            )
            return tok, outs

        # Use a small process pool sized to CPU count
        with ProcessPoolExecutor() as pool:
            tasks = [loop.run_in_executor(pool, submit_one, tok) for tok in tokens]
            results_per_token = await asyncio.gather(*tasks)

        # Store outputs via the registry
        for _tok, outputs in results_per_token:
            for name, df, update_mode, storage_period in outputs:
                # Defaults if not provided
                mode = update_mode or "overwrite"
                keep = storage_period if storage_period is not None else 0
                await self._registry.store_df(name, df, mode, keep)

        # All done
        return None
