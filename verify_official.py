import sys, json, types
from pathlib import Path
import numpy as np

def load_official(path):
    sys.modules.setdefault("heyoka", types.ModuleType("heyoka"))
    src = Path(path).read_text().replace("udp = celestial_morse_code()", "")
    ns = {"__file__": str(path)}
    exec(compile(src, str(path), "exec"), ns)
    cls = ns["celestial_morse_code"]
    mu_e, mu_m, LU = 3.986004418e14, 4.9048695e12, 389703e3
    TU = 1 / np.sqrt((mu_e + mu_m) / LU ** 3)
    cls._create_ta = lambda self: (None, TU, LU)
    data = Path(__file__).parent / "data"
    cls._get_file_path = lambda self, p: data / Path(p).name
    return cls()

if __name__ == "__main__":
    sol = json.load(open(sys.argv[1]))
    udp = load_official(sys.argv[2])
    obj, n, mse = udp.fitness(np.array(sol["x"]), postprocess=True)
    print(f"OFFICIAL  objective={obj[0]}  spacecraft={n}  MSE={mse:.6f}  feasible={mse <= udp.mse_thresh}")
