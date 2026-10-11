"""SmartPath vs conventional signal control: discrete-time (1 s) queue simulation.
Poisson arrivals, 30 replications, common random numbers across controllers.
Usage: python smartpath_sim.py  -> writes results.json
"""
import json, math, random
import numpy as np

FLOWS = np.array([700.0, 600.0, 300.0]) / 3600.0   # veh/s, TL1 TL2 TL3 (peak survey)
WEB_PEAK = [85, 73, 37]                              # Webster greens at peak (Table III)
YELLOW, ALLRED = 3, 1
SAT = 0.5                                            # veh/s (1800 veh/h)
STARTUP = 2                                          # s of start-up lost time at start of each green
END_GAIN = 2                                         # s of discharge continuing into the yellow
LOW_MAX, CUT = 10, 10
RUN = 3600
SCEN = {1: ("Peak", 1.0), 2: ("Moderate", 0.7), 3: ("Off-peak", 0.4), 4: ("Surge", 0.7)}

def arrivals(scen, rng):
    scale = SCEN[scen][1]
    lam = np.tile(FLOWS * scale, (RUN, 1))
    if scen == 4:
        lam[300:480, 0] *= 1.8
    return rng.poisson(lam)

def webster(scale, mingreen=8, mincycle=40):
    f = FLOWS * scale * 3600
    y = f / 1800.0; Y = y.sum(); L = 12
    C = max(mincycle, (1.5 * L + 5) / (1 - Y)); G = C - L
    return [max(mingreen, round(G * yi / Y)) for yi in y]

class Sim:
    def __init__(self, arr):
        self.arr = arr
        self.q = [0, 0, 0]; self.cap = 0.0
        self.t = 0; self.served = 0; self.arrived = 0
        self.dl = [0, 0, 0]; self.arrk = [0, 0, 0]
        self.qtot = np.zeros(RUN); self.peak = 0
        self.starts = []; self.last_arr = [-99, -99, -99]
    def tick(self, green=None, since_green=0):
        """advance one second; green = approach with green (or None)"""
        if self.t >= RUN: return False
        for k in range(3):
            n = int(self.arr[self.t, k])
            if n: self.q[k] += n; self.arrived += n; self.arrk[k] += n; self.last_arr[k] = self.t
        if green is not None and since_green >= STARTUP:
            self.cap = min(self.cap + SAT, 1.0 + SAT)
            while self.cap >= 1 and self.q[green] > 0:
                self.q[green] -= 1; self.cap -= 1; self.served += 1
            if self.q[green] == 0: self.cap = min(self.cap, 1.0)
        else:
            self.cap = 0.0 if green is None else self.cap
        for k in range(3): self.dl[k] += self.q[k]
        s = sum(self.q); self.qtot[self.t] = s; self.peak = max(self.peak, s)
        self.t += 1
        return self.t < RUN
    def clear(self, n):
        for _ in range(n):
            if not self.tick(None): return False
        return True
    def green(self, k, g):
        self.cap = 0.0
        for s in range(g):
            if not self.tick(k, s): return False
        return True

def measure(q, mode, rng):
    if mode == "camera":      # recall 1.00, precision 0.73 -> false alarms = 0.37 x true count
        return q + rng.poisson(0.37 * q)
    return q

def run_fixed(arr, greens):
    s = Sim(arr)
    while True:
        s.starts.append(s.t)
        for k in range(3):
            if not s.green(k, greens[k] + END_GAIN): return s
            if not s.clear(YELLOW + ALLRED - END_GAIN): return s

def run_smart(arr, sampling="own", noise=None, rng=None):
    s = Sim(arr); plan = list(WEB_PEAK)
    def g(k, q): return WEB_PEAK[k] if measure(q, noise, rng) > LOW_MAX else WEB_PEAK[k] - CUT
    while True:
        s.starts.append(s.t)
        if sampling == "cycle": plan = [g(k, s.q[k]) for k in range(3)]
        for k in range(3):
            gk = g(k, s.q[k]) if sampling == "own" else plan[k]
            if not s.green(k, gk + END_GAIN): return s
            if not s.clear(YELLOW + ALLRED - END_GAIN): return s

def run_actuated(arr, gmin=8, gap=3):
    s = Sim(arr); k = 0
    while True:
        if k == 0: s.starts.append(s.t)
        if s.q[k] == 0:                      # no call: skip phase
            k = (k + 1) % 3
            if all(q == 0 for q in s.q):
                if not s.clear(1): return s
            continue
        s.cap = 0.0; sg = 0
        while True:
            if not s.tick(k, sg): return s
            sg += 1
            if sg >= WEB_PEAK[k]: break
            if sg >= gmin and s.q[k] == 0 and s.t - 1 - s.last_arr[k] >= gap: break
        for e in range(END_GAIN):
            if not s.tick(k, sg + e): return s
        if not s.clear(YELLOW + ALLRED - END_GAIN): return s
        k = (k + 1) % 3

CONTROLLERS = ["Webster peak plan", "Webster re-timed", "Vehicle-actuated", "SmartPath", "SmartPath (cycle-start)", "SmartPath (camera error)"]

def metrics(s):
    cyc = np.diff(s.starts) if len(s.starts) > 1 else [np.nan]
    return dict(delay=sum(s.dl) / s.arrived, bway=(s.dl[0] + s.dl[1]) / (s.arrk[0] + s.arrk[1]),
                cycle=float(np.mean(cyc)), avgq=float(s.qtot.mean()), peak=s.peak,
                served=s.served, arrived=s.arrived, left=sum(s.q), qtot=s.qtot)

def main(N=30):
    out = {}
    for sc in SCEN:
        base_scale = 0.7 if sc == 4 else SCEN[sc][1]
        gre = webster(base_scale)
        res = {c: [] for c in CONTROLLERS}
        for r in range(N):
            rng = np.random.default_rng(1000 * sc + r)
            arr = arrivals(sc, rng)
            nrng = np.random.default_rng(777 + r)
            res["Webster peak plan"].append(metrics(run_fixed(arr, WEB_PEAK)))
            res["Webster re-timed"].append(metrics(run_fixed(arr, gre)))
            res["Vehicle-actuated"].append(metrics(run_actuated(arr)))
            res["SmartPath"].append(metrics(run_smart(arr)))
            res["SmartPath (cycle-start)"].append(metrics(run_smart(arr, "cycle")))
            res["SmartPath (camera error)"].append(metrics(run_smart(arr, "own", "camera", nrng)))
        out[sc] = dict(retimed=gre, res=res)
    return out

def ci(x):
    x = np.asarray(x, float); n = len(x)
    t = {29: 2.045}.get(n - 1, 1.96)
    return float(x.mean()), float(t * x.std(ddof=1) / math.sqrt(n))

if __name__ == "__main__":
    out = main()
    summ = {}
    for sc, d in out.items():
        summ[sc] = {"retimed": d["retimed"]}
        for c, rs in d["res"].items():
            row = {}
            for key in ["delay", "bway", "cycle", "avgq", "peak", "served", "arrived", "left"]:
                row[key] = ci([r[key] for r in rs])
            q = np.mean([r["qtot"] for r in rs], axis=0)
            row["run_avg"] = [float(q[:600 * m].mean()) for m in range(1, 7)]
            row["run_avg_1_6"] = [float(q[:60 * m].mean()) for m in range(1, 7)]
            row["win5"] = [float(q[i * 300:(i + 1) * 300].mean()) for i in range(12)]
            base = [r["delay"] for r in d["res"]["Webster peak plan"]]
            ret = [r["delay"] for r in d["res"]["Webster re-timed"]]
            row["d_vs_peak"] = ci([(b - r["delay"]) / b * 100 for b, r in zip(base, rs)])
            row["d_vs_ret"] = ci([(b - r["delay"]) / b * 100 for b, r in zip(ret, rs)])
            summ[sc][c] = row
    json.dump(summ, open("results.json", "w"), indent=1)
    for sc in summ:
        print(SCEN[sc][0], summ[sc]["retimed"])
        for c in CONTROLLERS:
            r = summ[sc][c]
            print("  %-26s delay %5.1f±%.1f bway %5.1f cyc %5.0f q %4.1f peak %4.1f served %6.1f/%6.1f  vsPeak %5.1f±%.1f vsRet %6.1f"%(c, *r["delay"], r["bway"][0], r["cycle"][0], r["avgq"][0], r["peak"][0], r["served"][0], r["arrived"][0], *r["d_vs_peak"], r["d_vs_ret"][0]))
