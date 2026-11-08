# Q2b - Penjadwalan ujian sebagai CSP (Backtracking + MRV + Forward Checking)
students = {                      # mata kuliah -> mahasiswa peserta
    "AI":      {"s1", "s2", "s3", "s4"},
    "BasisData": {"s2", "s5", "s6"},
    "Jarkom":  {"s3", "s7"},
    "PBO":     {"s1", "s8", "s9"},
    "Statistik": {"s4", "s5", "s10"},
}
slots  = ["Sen-08", "Sen-10", "Sel-08"]
rooms  = {"R101": 40, "R102": 5}                  # kapasitas ruangan
domains = {e: [(t, r) for t in slots for r, cap in rooms.items() if cap >= len(students[e])]
           for e in students}

def conflict(e1, v1, e2, v2):
    (t1, r1), (t2, r2) = v1, v2
    if t1 == t2 and students[e1] & students[e2]: return True   # mahasiswa bentrok
    if t1 == t2 and r1 == r2: return True                      # ruangan sama
    return False

def backtrack(assign, doms):
    if len(assign) == len(students): return assign
    e = min((x for x in students if x not in assign), key=lambda x: len(doms[x]))  # MRV
    for v in doms[e]:
        if all(not conflict(e, v, o, assign[o]) for o in assign):
            new = {k: [d for d in dv if not conflict(e, v, k, d)] if k not in assign and k != e else dv
                   for k, dv in doms.items()}                   # forward checking
            if all(new[k] for k in new if k not in assign and k != e):
                res = backtrack({**assign, e: v}, new)
                if res: return res
    return None

sol = backtrack({}, domains)
for e, (t, r) in sorted(sol.items(), key=lambda x: x[1]):
    print(f"{e:10s} -> slot {t}, ruang {r}")
