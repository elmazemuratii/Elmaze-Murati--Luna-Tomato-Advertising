
from pathlib import Path
import re
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"

MORSE = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.',
    'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.',
    'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-',
    'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--', 'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
    '6': '-....', '7': '--...', '8': '---..', '9': '----.', ' ': '/',
}


class Problem:
    def __init__(self, message=" Tomatoes for sale ", data_dir=DATA):
        self.message = message
        # --- CR3BP constants (identical to the official UDP) ---
        mu_e, mu_m = 3.986004418e14, 4.9048695e12
        self.LU = 389703e3
        n = np.sqrt((mu_e + mu_m) / self.LU ** 3)
        self.TU = 1.0 / n
        self.mu = mu_m / (mu_e + mu_m)
        self.r_moon = 1737.4e3 / self.LU
        self.synodic_period = self.TU / 86400 * 2 * np.pi            # days

        # --- Morse timing scaled so that the message spans 0.9 synodic periods ---
        dot, dash, gap, lgap, wgap = 2, 5, 2, 5, 7
        L = self._morse_duration(message, dot, dash, gap, lgap, wgap)
        sf = 0.9 * self.synodic_period / L
        self.dot, self.dash, self.gap, self.letter_gap, self.word_gap = [v * sf for v in (dot, dash, gap, lgap, wgap)]
        self.width = self.dot / 2
        self.dt = self.dot / 20
        self.t, self.target = self._signal(message)

        # --- Orbit databases ---
        self.db = {k: np.loadtxt(Path(data_dir) / f"db_{k}.txt", delimiter=",", skiprows=1)
                   for k in ("dro", "lyap", "axial")}
        self.n_orb = {k: len(v) for k, v in self.db.items()}
        self.num_orbits = sum(self.n_orb.values())
        self.max_per_orbit = 5
        self.mse_thresh = 0.05
        # frequencies (columns 7 / 8, as in the official code)
        self.f1 = {k: v[:, 7] for k, v in self.db.items()}
        self.f2 = {k: v[:, 8] for k, v in self.db.items()}
        self.n_box = {"dro": 1, "lyap": 2, "axial": 2}
        # flat orbit table: (family, index within family)
        self.orbits = [(k, i) for k in ("dro", "lyap", "axial") for i in range(self.n_orb[k])]
        self.offset = {"dro": 0, "lyap": self.n_orb["dro"], "axial": self.n_orb["dro"] + self.n_orb["lyap"]}

    # ---------------------------------------------------------------- Morse
    @staticmethod
    def _morse_string(msg):
        return ' '.join(MORSE[c] for c in msg.upper())

    def morse_string(self):
        return self._morse_string(self.message)

    def _morse_duration(self, msg, dot, dash, gap, lgap, wgap):
        s = re.sub(r"\s*/\s*", " / ", self._morse_string(msg).strip())
        s = re.sub(r"\s{3,}", " / ", s)
        total = 0
        words = [w.strip() for w in s.split("/") if w.strip()]
        for wi, w in enumerate(words):
            letters = [l for l in w.split(" ") if l]
            for li, l in enumerate(letters):
                for si, sym in enumerate(l):
                    total += dot if sym == "." else dash
                    if si < len(l) - 1:
                        total += gap
                if li < len(letters) - 1:
                    total += lgap
            if wi < len(words) - 1:
                total += wgap
        return total

    def _signal(self, msg):
        t, segs = 0.0, []
        for ch in msg.upper():
            if ch == ' ':
                t += self.word_gap
                continue
            for sym in MORSE[ch]:
                d = self.dot if sym == '.' else self.dash
                segs.append((t, t + d))
                t += d + self.gap
            t += self.letter_gap - self.gap
        tg = np.arange(0, t, self.dt)
        s = np.zeros_like(tg)
        for a, b in segs:
            s[int(round(a / self.dt)):int(round(b / self.dt))] = 1.0
        return tg, s

    # ------------------------------------------------------- occultation boxes
    def _box(self, centre_nd, freq):
        """Repeating box window (exact copy of the official formula), times in nd units."""
        P = 1.0 / freq
        tn = self.t * 86400 / self.TU
        wn = self.width * 86400 / self.TU
        ph = (tn - centre_nd + P / 2) % P - P / 2
        k = 1e6
        return np.minimum(1, np.maximum(0, k * (ph + wn / 2))) - np.minimum(1, np.maximum(0, k * (ph - wn / 2)))

    def spacecraft_signal(self, fam, i, ph):
        """On-mask contribution of ONE spacecraft (orbit i of family fam, phase ph)."""
        f1, f2 = self.f1[fam][i], self.f2[fam][i]
        s = self._box((2 * np.pi - ph) / (2 * np.pi) / f1, f1)
        if fam != "dro":
            s = s + self._box((ph + np.pi) / f2, f2)
        return s

    # --------------------------------------------------------------- fitness
    def decode(self, x):
        x = np.asarray(x, float)
        N = self.num_orbits
        sel = np.maximum(np.round(x[:N]).astype(int), 0)
        return sel, x[N:]

    def fleet(self, x):
        """[(flat_orbit_index, family, i, phase), ...] for every active spacecraft."""
        sel, phs = self.decode(x)
        out = []
        for flat, (fam, i) in enumerate(self.orbits):
            for j in range(sel[flat]):
                out.append((flat, fam, i, phs[flat * self.max_per_orbit + j]))
        return out

    def reconstruct(self, x):
        rec = np.zeros_like(self.target)
        for _, fam, i, ph in self.fleet(x):
            rec += self.spacecraft_signal(fam, i, ph)
        return np.clip(rec, 0, 1)

    def fitness(self, x):
        """Returns (objective, num_spacecraft, mse) - identical to udp.fitness(x, postprocess=True)."""
        mse = float(np.mean((self.reconstruct(x) - self.target) ** 2))
        n = int(np.sum(self.decode(x)[0]))
        if mse > self.mse_thresh:
            obj = self.max_per_orbit * self.num_orbits + 1e3 * (np.sqrt(mse) - np.sqrt(self.mse_thresh))
        else:
            obj = n
        return obj, n, mse

    def encode(self, fleet):
        """fleet: list of (flat_orbit_index, phase) -> chromosome x."""
        N = self.num_orbits
        x = np.zeros(N + N * self.max_per_orbit)
        cnt = np.zeros(N, int)
        for flat, ph in fleet:
            assert cnt[flat] < self.max_per_orbit
            x[N + flat * self.max_per_orbit + cnt[flat]] = ph
            cnt[flat] += 1
        x[:N] = cnt
        return x
