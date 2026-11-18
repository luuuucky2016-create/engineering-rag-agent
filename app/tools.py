"""Agent tools: a safe calculator and a knowledge-base search over the vector store."""

from __future__ import annotations

import ast
import math
import operator as op

from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import VectorStore

from app.rag import format_results

_BIN_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
}
_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}
_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "round": round,
    "min": min, "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic without eval(): only numbers, + - * / // % **, whitelisted functions/constants."""

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            right = _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](_eval(node.left), right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
            return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
        raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")

    return _eval(ast.parse(expression.replace("^", "**"), mode="eval"))


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression exactly, e.g. '2 * 18 * 25 * 0.00727' or 'sqrt(3) * 400 * 32'.
    Supports + - * / // % ** (or ^), sqrt, log, log10, exp, sin, cos, tan, abs, round, min, max, pi, e.
    Always use this for arithmetic instead of computing in your head."""
    try:
        result = safe_eval(expression)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as exc:  # report back to the model instead of crashing the graph
        return f"Error: {exc}"
    return f"{result:.6g}" if isinstance(result, float) else str(result)


# IEC 60228 class 2 copper, max DC resistance at 20 °C (Ω/km) - same table as data/docs/voltage_drop.md
COPPER_R_OHM_PER_KM = {
    1.5: 12.1, 2.5: 7.41, 4: 4.61, 6: 3.08, 10: 1.83, 16: 1.15,
    25: 0.727, 35: 0.524, 50: 0.387, 70: 0.268, 95: 0.193, 120: 0.153,
}


@tool
def voltage_drop(current_a: float, length_m: float, cable_mm2: float, supply_v: float, phases: int = 3) -> str:
    """Compute cable voltage drop for a copper cable run. Use this for ANY voltage-drop question.
    current_a: load current in amperes; length_m: one-way cable length in metres;
    cable_mm2: conductor size in mm² (e.g. 2.5, 6, 10); supply_v: supply voltage (line-to-line for
    three-phase, e.g. 400); phases: 1 or 3."""
    r = COPPER_R_OHM_PER_KM.get(float(cable_mm2))
    if r is None:
        return f"Error: no resistance data for {cable_mm2} mm² (available: {sorted(COPPER_R_OHM_PER_KM)})"
    if phases not in (1, 3):
        return "Error: phases must be 1 or 3"
    k, k_txt = (math.sqrt(3), "√3") if phases == 3 else (2.0, "2")
    vd = k * current_a * length_m * r / 1000
    pct = vd / supply_v * 100
    if pct <= 3:
        verdict = "within the common 3% final-circuit guideline (and the 5% total guideline)"
    elif pct <= 5:
        verdict = "above the 3% final-circuit guideline but within the 5% total guideline"
    else:
        verdict = "above the common 5% total guideline - consider a larger cable"
    return (
        f"Voltage drop = {vd:.2f} V = {pct:.2f}% of {supply_v:g} V; {verdict}. "
        f"Formula: {k_txt} x I x L x R / 1000 with I={current_a:g} A, L={length_m:g} m, "
        f"R={r} ohm/km ({cable_mm2:g} mm² copper, IEC 60228)."
    )


def make_kb_search_tool(store: VectorStore, k: int = 4, min_relevance: float = 0.0) -> BaseTool:
    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the engineering knowledge base. Use for any question about electrical formulas,
        voltage drop limits, cable sizing, IP ratings or other reference material. Returns passages
        tagged with [source: file#chunk] - cite these in your answer. If it returns NO_RESULTS the
        topic is not covered and you must say so."""
        hits = store.similarity_search_with_relevance_scores(query, k=k)
        return format_results([d for d, s in hits if s >= min_relevance], [s for _, s in hits if s >= min_relevance])

    return search_knowledge_base
