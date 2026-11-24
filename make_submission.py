"""Turns results/best_fleet.json into results/solution.json (and optionally uploads via ESA's helper)."""
import json, sys
from lta.problem import Problem
from lta.solver import save_solution

fleet = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/best_fleet.json"))
p = Problem()
x = p.encode([(f, ph) for f, ph in fleet])
print("exact fitness (objective, K, mse):", save_solution(p, x, "results/solution.json"))
# Upload (needs ESA's `_submission_helper` and your credentials):
#   from _submission_helper import create_submission
#   create_submission("spoc-4-luna-tomato-advertising", "tie-breaker", list(x), "results/solution_submit.json")
