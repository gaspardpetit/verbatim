"""Parse diarization policy strings into strategy assignments."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class PolicyClause:
    targets: Set[int]  # empty set means wildcard
    strategy: str
    params: Dict[str, str]


def parse_targets(target_str: str) -> Set[int]:
    target_str = target_str.strip()
    if target_str == "*":
        return set()
    targets: Set[int] = set()
    for part in target_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                start, end = end, start
            targets.update(range(start, end + 1))
        else:
            targets.add(int(part))
    return targets


def parse_params(param_str: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for pair in param_str.split("&"):
        if not pair:
            continue
        if "=" in pair:
            k, v = pair.split("=", 1)
            params[k.strip()] = v.strip()
        else:
            params[pair.strip()] = ""
    return params


def parse_clause(clause: str) -> PolicyClause:
    clause = clause.strip()
    target_part, rest = clause.split("=", 1) if "=" in clause else ("*", clause)
    if "?" in rest:
        strategy_part, param_part = rest.split("?", 1)
        params = parse_params(param_part)
    else:
        strategy_part = rest
        params = {}
    targets = parse_targets(target_part)
    strategy = strategy_part.strip()
    return PolicyClause(targets=targets, strategy=strategy, params=params)


def parse_policy(policy: str) -> List[PolicyClause]:
    if not policy:
        return []
    clauses = []
    for raw in policy.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        clauses.append(parse_clause(raw))
    return clauses


def assign_channels(clauses: List[PolicyClause], nchannels: int) -> Dict[int, PolicyClause]:
    assignments: Dict[int, PolicyClause] = {}
    wildcard: Optional[PolicyClause] = None
    for clause in clauses:
        if len(clause.targets) == 0:
            wildcard = clause
            continue
        for ch in clause.targets:
            assignments[ch] = clause
    if wildcard:
        for ch in range(nchannels):
            if ch not in assignments:
                assignments[ch] = wildcard
    return assignments
