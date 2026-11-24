import time, json
import numpy as np
from .problem import Problem

BOXLEN = 10   # width / dt == 10 samples (exact)


class Engine:
    def __init__(self, prob: Problem, m_dro=8, m_lyap=32768, seed=0):
        self.p = prob
        self.n = len(prob.t)
        self.dt, self.w, self.T = prob.dt, prob.width, prob.t[-1] + prob.dt
        self.tgt = prob.target.astype(np.int8)
        self.rng = np.random.default_rng(seed)
        self.max_err = int(np.floor(prob.mse_thresh * self.n))
        self.m_dro, self.m_lyap = m_dro, m_lyap
        tu_d = prob.TU / 86400.0
        # period in days for every flat orbit; train periods (days)
        self.Pd = np.array([1.0 / prob.f1[f][i] * tu_d for f, i in prob.orbits])
        self.Pnd = np.array([1.0 / prob.f1[f][i] for f, i in prob.orbits])
        self.tu_d = tu_d
        self.fam = np.array([prob.n_box[f] for f, _ in prob.orbits])      # 1 = dro, 2 = lyap/axial
        self.cnt = np.zeros(self.n, np.int32)
        self.fleet = []     # list of (flat, phase)

    # ---------------------------------------------------------------- masks
    def _train_idx(self, c, P):
        """sample start indices of all boxes of a train (c: array of centres in days)"""
        kmax = int(self.T // P) + 2
        k = np.arange(-1, kmax)[:, None]
        a = np.ceil((c[None, :] + k * P - self.w / 2) / self.dt).astype(np.int64)
        return a                                                  # (K, M)

    def centres(self, flat, ph):
        """train centres in days for phase array ph -> list of (c_days, P_days)"""
        ph = np.atleast_1d(ph)
        P = self.Pnd[flat]
        c1 = ((2 * np.pi - ph) / (2 * np.pi) * P) % P
        out = [(c1 * self.tu_d, self.Pd[flat])]
        if self.fam[flat] == 2:
            c2 = ((ph + np.pi) * P) % P
            out.append((c2 * self.tu_d, self.Pd[flat]))
        return out

    def mask_idx(self, flat, ph):
        """integer array of sample indices covered by one spacecraft"""
        idx = []
        for c, P in self.centres(flat, ph):
            a = self._train_idx(c, P).ravel()
            for o in range(BOXLEN):
                j = a + o
                idx.append(j[(j >= 0) & (j < self.n)])
        return np.concatenate(idx)

    # ---------------------------------------------------------------- state
    def error(self):
        return int(np.sum((self.cnt > 0) != (self.tgt > 0)))

    def add(self, flat, ph):
        np.add.at(self.cnt, self.mask_idx(flat, ph), 1)
        self.fleet.append((int(flat), float(ph)))

    def remove(self, k):
        flat, ph = self.fleet.pop(k)
        np.add.at(self.cnt, self.mask_idx(flat, ph), -1)
        return flat, ph

    def rebuild(self, fleet):
        self.cnt[:] = 0
        self.fleet = []
        for flat, ph in fleet:
            self.add(flat, ph)

    # ----------------------------------------------------------------- cache
    OFF = 4000

    def build_cache(self):
        """pre-compute, for every orbit, the box start indices for every phase of its grid"""
        self.PH, self.A = {}, {}
        for flat in range(self.p.num_orbits):
            ph = self.phase_grid(flat)
            self.PH[flat] = ph
            self.A[flat] = [(self._train_idx(c, P) + self.OFF).astype(np.int16) for c, P in self.centres(flat, ph)]

    def gain_cumsum_pad(self, lam=1.0):
        on = self.tgt > 0
        wgt = (self.cnt == 0) * np.where(on, 1.0, -lam)
        cs = np.concatenate([[0], np.cumsum(wgt)])
        return np.concatenate([np.zeros(self.OFF), cs, np.full(2 * self.OFF + 40, cs[-1])])

    def scan_fast(self, orbit_ids, csp, used):
        best = (-1e18, -1, 0.0)
        for flat in orbit_ids:
            if used[flat] >= self.p.max_per_orbit:
                continue
            g = 0
            for a in self.A[flat]:
                g = g + (csp[a + BOXLEN] - csp[a]).sum(0)
            j = int(np.argmax(g))
            if g[j] > best[0]:
                best = (float(g[j]), flat, float(self.PH[flat][j]))
        return best

    # ----------------------------------------------------------------- scan
    def phase_grid(self, flat):
        if self.fam[flat] == 1:
            m = int(self.m_dro * self.Pd[flat] / self.dt) + 1
        else:
            m = self.m_lyap
        return np.linspace(0, 2 * np.pi, m, endpoint=False)

    def scan(self, orbit_ids, cs, used=None):
        """best (gain, flat, phase) over the given orbits.  gain = reduction of the error"""
        best = (-10**9, -1, 0.0)
        for flat in orbit_ids:
            if used is not None and used[flat] >= self.p.max_per_orbit:
                continue
            ph = self.phase_grid(flat)
            g = np.zeros(len(ph))
            for c, P in self.centres(flat, ph):
                a = self._train_idx(c, P)
                lo = np.clip(a, 0, self.n)
                hi = np.clip(a + BOXLEN, 0, self.n)
                g += (cs[hi] - cs[lo]).sum(0)
            j = int(np.argmax(g))
            if g[j] > best[0]:
                best = (int(g[j]), flat, float(ph[j]))
        return best

    def gain_cumsum(self):
        wgt = (self.cnt == 0) * (2 * self.tgt.astype(np.int64) - 1)
        return np.concatenate([[0], np.cumsum(wgt)])

    def used(self):
        u = np.zeros(self.p.num_orbits, int)
        for f, _ in self.fleet:
            u[f] += 1
        return u

    # --------------------------------------------------------------- greedy
    def greedy(self, subset=None, verbose=True):
        ids = np.arange(self.p.num_orbits)
        while self.error() > self.max_err:
            g, flat, ph = self.scan(ids, self.gain_cumsum(), self.used())
            if g <= 0:
                break
            self.add(flat, ph)
            if verbose:
                print(f"  greedy K={len(self.fleet):3d} err={self.error()}", flush=True)

    # ------------------------------------------------------------ LNS / SA
    def lns(self, time_limit=60, n_sub=120, temp=0.6, verbose=True):
        """Try to reach error <= max_err with the CURRENT number of spacecraft."""
        t0 = time.time()
        E = self.error()
        it = 0
        while E > self.max_err and time.time() - t0 < time_limit:
            it += 1
            r = int(self.rng.choice([1, 1, 1, 2, 2, 3]))
            removed = [self.remove(int(self.rng.integers(len(self.fleet)))) for _ in range(min(r, len(self.fleet)))]
            for _ in removed:
                sub = self.rng.choice(self.p.num_orbits, n_sub, replace=False)
                sub = np.unique(np.concatenate([sub, np.arange(self.p.n_orb['dro'])]))
                g, flat, ph = self.scan(sub, self.gain_cumsum(), self.used())
                self.add(flat, ph)
            E2 = self.error()
            if E2 <= E or self.rng.random() < np.exp(-(E2 - E) / temp):
                E = E2
            else:                       # revert
                for _ in removed:
                    self.remove(len(self.fleet) - 1)
                for flat, ph in removed:
                    self.add(flat, ph)
            if verbose and it % 20 == 0:
                print(f"    K={len(self.fleet)} it={it} err={E} (target<={self.max_err})", flush=True)
        return E

    def lns_fast(self, time_limit=60, n_sub=150, temp=0.7, log=None):
        """LNS / simulated annealing at fixed K using the cached scan. Stops when error <= max_err."""
        t0, it = time.time(), 0
        E = self.error()
        best = (E, list(self.fleet))
        dro = np.arange(self.p.n_orb['dro'])
        while E > self.max_err and time.time() - t0 < time_limit:
            it += 1
            r = int(self.rng.choice([1, 1, 2, 2, 3]))
            removed = [self.remove(int(self.rng.integers(len(self.fleet)))) for _ in range(min(r, len(self.fleet)))]
            for _ in removed:
                sub = np.concatenate([self.rng.choice(self.p.num_orbits, n_sub, replace=False), dro[self.rng.random(len(dro)) < 0.5]])
                g, flat, ph = self.scan_fast(sub, self.gain_cumsum_pad(1.0), self.used())
                self.add(flat, ph)
            E2 = self.error()
            if E2 <= E or self.rng.random() < np.exp(-(E2 - E) / temp):
                E = E2
                if E < best[0]:
                    best = (E, list(self.fleet))
            else:
                for _ in removed:
                    self.remove(len(self.fleet) - 1)
                for flat, ph in removed:
                    self.add(flat, ph)
            if log and it % 25 == 0:
                log(f"K={len(self.fleet)} it={it} err={E} best={best[0]} t={time.time()-t0:.0f}s")
        if best[0] < E:
            self.rebuild(best[1]); E = best[0]
        return E

    # --------------------------------------------------------------- shrink
    def drop_best(self):
        """remove the spacecraft whose loss hurts the least"""
        best, bk = 10**9, 0
        for k in range(len(self.fleet)):
            fl, ph = self.fleet[k]
            self.remove(k)
            e = self.error()
            self.fleet.insert(k, (fl, ph)); np.add.at(self.cnt, self.mask_idx(fl, ph), 1)
            if e < best:
                best, bk = e, k
        self.remove(bk)
        return best

    # --------------------------------------------------------------- polish
    def polish(self, iters=4000, sigma=2e-3):
        """continuous refinement of every phase (hill-climbing on the error)"""
        E = self.error()
        for _ in range(iters):
            k = int(self.rng.integers(len(self.fleet)))
            fl, ph = self.fleet[k]
            new = (ph + self.rng.normal(0, sigma * self.rng.choice([1, 0.1, 0.01, 10]))) % (2 * np.pi)
            self.remove(k); self.add(fl, new)
            E2 = self.error()
            if E2 <= E:
                E = E2
                self.fleet[-1] = (fl, new)
            else:
                self.remove(len(self.fleet) - 1); self.add(fl, ph)
            # keep list order irrelevant
        return E

    def chromosome(self):
        return self.p.encode(self.fleet)


def save_solution(prob, x, path, name="solution"):
    obj, n, mse = prob.fitness(x)
    json.dump({"name": name, "num_spacecraft": n, "mse": mse, "objective": obj,
               "x": [float(v) for v in x]}, open(path, "w"))
    return obj, n, mse


def run_pipeline(prob, total_time=600, per_k=60, seed=0, ckpt="results/best_fleet.json", log=print,
                 m_dro=4, m_lyap=16384, start_fleet=None):
    """greedy -> LNS -> (shrink -> LNS)*  ; keeps the best feasible fleet in `ckpt`."""
    t0 = time.time()
    e = Engine(prob, m_dro=m_dro, m_lyap=m_lyap, seed=seed)
    e.build_cache()
    ids = np.arange(prob.num_orbits)
    if start_fleet:
        e.rebuild([tuple(f) for f in start_fleet])
    else:
        while e.error() > e.max_err:
            g, fl, ph = e.scan_fast(ids, e.gain_cumsum_pad(1.0), e.used())
            if g <= 0:
                break
            e.add(fl, ph)
    log(f"greedy: K={len(e.fleet)} err={e.error()}")
    E = e.lns_fast(time_limit=per_k * 2, log=log)
    while E > e.max_err:                       # not feasible yet -> allow one more spacecraft
        g, fl, ph = e.scan_fast(ids, e.gain_cumsum_pad(1.0), e.used())
        e.add(fl, ph)
        log(f"adding spacecraft -> K={len(e.fleet)}")
        E = e.lns_fast(time_limit=per_k * 2, log=log)
    best = list(e.fleet)
    json.dump(best, open(ckpt, "w"))
    while time.time() - t0 < total_time:
        e.drop_best()
        E = e.lns_fast(time_limit=min(per_k, total_time - (time.time() - t0)), log=log)
        if E <= e.max_err:
            best = list(e.fleet)
            json.dump(best, open(ckpt, "w"))
            log(f"*** feasible with K={len(best)} (err={E}) t={time.time()-t0:.0f}s")
        else:
            log(f"failed at K={len(e.fleet)} -> restore K={len(best)} and perturb")
            e.rebuild(best)
            # perturb: remove two random, re-add via scan, then continue shrinking
            for _ in range(2):
                e.remove(int(e.rng.integers(len(e.fleet))))
            e.lns_fast(time_limit=per_k / 2)
            if e.error() > e.max_err:
                e.rebuild(best)
    return best
