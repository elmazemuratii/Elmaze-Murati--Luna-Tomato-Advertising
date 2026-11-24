"""PyCharm entry point:  python run_solver.py --time 900 --seed 0"""
import argparse, json
from lta.problem import Problem
from lta.solver import run_pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=float, default=600, help="wall-clock budget in seconds")
    ap.add_argument("--per-k", type=float, default=60, help="LNS seconds per spacecraft count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default=None, help="json fleet to resume from")
    a = ap.parse_args()
    prob = Problem()
    start = json.load(open(a.start)) if a.start else None
    best = run_pipeline(prob, a.time, a.per_k, a.seed, "results/best_fleet.json", start_fleet=start)
    print("best K =", len(best))
