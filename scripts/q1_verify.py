from fractions import Fraction
from itertools import product

import numpy as np
from tqdm import tqdm

from src import config
from src.utils import make_dirs, save_json, set_seed

TRENDS = [-1, 0, 1]
ONE_HOT = {-1: (1, 0, 0), 0: (0, 1, 0), 1: (0, 0, 1)}


def all_count_triples(k):
    return [(a, b, c) for a in range(k + 1) for b in range(k + 1) for c in range(k + 1) if 1 <= a + b + c <= k]


def triple_to_neighbourhood(triple):
    a, b, c = triple
    return [-1] * a + [0] * b + [1] * c


def part_a():
    first = [-1, 1]
    second = [0, 0]
    rows = []
    rng = np.random.default_rng(config.SEED)
    for w, c in rng.normal(size=(5, 2)):
        agg1 = sum(w * x + c for x in first)
        agg2 = sum(w * x + c for x in second)
        rows.append({"w": w, "c": c, "agg_first": agg1, "agg_second": agg2, "equal": bool(np.isclose(agg1, agg2))})
    return {
        "formula": "sum_u (w x_u + c) = w * S_v + c * n_v",
        "first": {"neighbours": first, "n": len(first), "S": sum(first), "steady": first.count(0)},
        "second": {"neighbours": second, "n": len(second), "S": sum(second), "steady": second.count(0)},
        "random_checks": rows,
        "all_equal": all(r["equal"] for r in rows),
    }


def part_b(k=8):
    base = k + 1
    phi = {-1: 1, 0: base, 1: base ** 2}
    triples = all_count_triples(k)
    sums = {}
    decoded_ok = True
    for triple in tqdm(triples, desc="Q1(b) checking triples"):
        total = sum(phi[x] for x in triple_to_neighbourhood(triple))
        sums.setdefault(total, []).append(triple)
        decoded = (total % base, (total // base) % base, total // base ** 2)
        decoded_ok &= decoded == triple
    collisions = {s: t for s, t in sums.items() if len(t) > 1}

    small_base = k
    small_phi = {-1: 1, 0: small_base, 1: small_base ** 2}
    small_sums = {}
    for triple in triples:
        total = sum(small_phi[x] for x in triple_to_neighbourhood(triple))
        small_sums.setdefault(total, []).append(triple)
    small_collisions = [t for t in small_sums.values() if len(t) > 1]

    nonsteady_ok = all(
        sum(x * x for x in triple_to_neighbourhood(t)) == t[0] + t[2] for t in triples
    )
    return {
        "K": k,
        "base": base,
        "phi": {"-1": phi[-1], "0": phi[0], "+1": phi[1]},
        "num_triples": len(triples),
        "num_distinct_sums": len(sums),
        "injective": len(collisions) == 0,
        "decoding_correct": bool(decoded_ok),
        "max_sum": max(sums),
        "base_K_collision_example": small_collisions[0] if small_collisions else None,
        "base_K_num_collisions": len(small_collisions),
        "nonsteady_phi": "phi(x) = x^2",
        "nonsteady_correct": bool(nonsteady_ok),
    }


def aggregate(neighbourhood, method):
    vectors = [ONE_HOT[x] for x in neighbourhood]
    if method == "sum":
        return tuple(sum(v[i] for v in vectors) for i in range(3))
    if method == "mean":
        return tuple(Fraction(sum(v[i] for v in vectors), len(vectors)) for i in range(3))
    return tuple(max(v[i] for v in vectors) for i in range(3))


TASKS = {
    "(i) count steady": lambda n: n.count(0),
    "(ii) more than half slowing": lambda n: n.count(-1) > len(n) / 2,
    "(iii) any slowing": lambda n: n.count(-1) > 0,
}

RULES = {
    ("(i) count steady", "sum"): ("coordinate 2", lambda v: v[1]),
    ("(ii) more than half slowing", "sum"): ("s1 > s2 + s3", lambda v: v[0] > v[1] + v[2]),
    ("(ii) more than half slowing", "mean"): ("m1 > 1/2", lambda v: v[0] > Fraction(1, 2)),
    ("(iii) any slowing", "sum"): ("s1 > 0", lambda v: v[0] > 0),
    ("(iii) any slowing", "mean"): ("m1 > 0", lambda v: v[0] > 0),
    ("(iii) any slowing", "max"): ("max1 = 1", lambda v: v[0] == 1),
}


def show(vector):
    return "(" + ", ".join(str(x) for x in vector) + ")"


def part_c(k=8):
    neighbourhoods = [triple_to_neighbourhood(t) for t in all_count_triples(k)]
    neighbourhoods.sort(key=len)
    rows = []
    for (task, answer_fn), method in tqdm(list(product(TASKS.items(), ["sum", "mean", "max"])), desc="Q1(c) table"):
        seen = {}
        counterexample = None
        for n in neighbourhoods:
            vector = aggregate(n, method)
            answer = answer_fn(n)
            if vector in seen and seen[vector][1] != answer and counterexample is None:
                counterexample = (seen[vector][0], n, vector, seen[vector][1], answer)
            seen.setdefault(vector, (n, answer))
        possible = counterexample is None
        row = {"task": task, "method": method, "possible": "Yes" if possible else "No"}
        if possible:
            rule_name, rule = RULES[(task, method)]
            verified = all(rule(aggregate(n, method)) == answer_fn(n) for n in neighbourhoods)
            row["reason"] = f"rho uses {rule_name}"
            row["rule_verified_on_all"] = verified
        else:
            first, second, vector, ans1, ans2 = counterexample
            row["reason"] = (f"{first} and {second} both give {show(vector)} "
                             f"but answers are {ans1} vs {ans2}")
            row["example"] = {"first": first, "second": second, "vector": show(vector),
                              "answer_first": ans1, "answer_second": ans2}
        rows.append(row)
    return rows


def write_table(rows):
    lines = ["| Task | Method | Possible? | Reason |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['method'].capitalize()} | {r['possible']} | {r['reason']} |")
    (config.TABLES_DIR / "q1c_table.md").write_text("\n".join(lines) + "\n")
    with open(config.TABLES_DIR / "q1c_table.csv", "w") as f:
        f.write("task,method,possible,reason\n")
        for r in rows:
            f.write(f"\"{r['task']}\",{r['method']},{r['possible']},\"{r['reason']}\"\n")


def main():
    set_seed()
    make_dirs()
    results = {"part_a": part_a(), "part_b": part_b(8), "part_c": part_c(8)}
    write_table(results["part_c"])
    save_json(results, config.RESULTS_DIR / "q1_results.json")
    b = results["part_b"]
    print(f"Q1(a) equal aggregates for all random (w, c): {results['part_a']['all_equal']}")
    print(f"Q1(b) K=8 base={b['base']} phi={b['phi']} triples={b['num_triples']} "
          f"injective={b['injective']} decoding={b['decoding_correct']} nonsteady={b['nonsteady_correct']}")
    print(f"Q1(b) base K collision example: {b['base_K_collision_example']}")
    for r in results["part_c"]:
        print(f"Q1(c) {r['task']:<30} {r['method']:<5} {r['possible']:<4} {r['reason']}")


if __name__ == "__main__":
    main()
