# Q3b - Q-Learning: robot pengantar obat di grid 5x5 dengan hambatan
import numpy as np, random
random.seed(1); np.random.seed(1)

N = 5
START, GOAL = (0, 0), (4, 4)
OBSTACLES = {(1, 1), (2, 3), (3, 1)}
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]        # atas, bawah, kiri, kanan
NAMES = ["^", "v", "<", ">"]
Q = np.zeros((N, N, len(ACTIONS)))
alpha, gamma, eps = 0.1, 0.95, 1.0

def step(s, a):
    r, c = s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1]
    if not (0 <= r < N and 0 <= c < N): return s, -5, False     # menabrak dinding
    if (r, c) in OBSTACLES:             return s, -20, False    # menabrak hambatan
    if (r, c) == GOAL:                  return (r, c), 100, True  # obat terkirim
    return (r, c), -1, False                                    # biaya langkah

for ep in range(3000):
    s, done = START, False
    while not done:
        a = random.randrange(4) if random.random() < eps else int(np.argmax(Q[s]))
        s2, r, done = step(s, a)
        Q[s][a] += alpha * (r + gamma * np.max(Q[s2]) * (not done) - Q[s][a])   # Bellman update
        s = s2
    eps = max(0.05, eps * 0.998)

print("Kebijakan optimal (S=start, G=goal, X=hambatan):")
for i in range(N):
    row = ""
    for j in range(N):
        if (i, j) == START: row += " S "
        elif (i, j) == GOAL: row += " G "
        elif (i, j) in OBSTACLES: row += " X "
        else: row += f" {NAMES[int(np.argmax(Q[i, j]))]} "
    print(row)

s, path = START, [START]
while s != GOAL and len(path) < 30:
    s, _, _ = step(s, int(np.argmax(Q[s]))); path.append(s)
print("Rute dipelajari:", path, "| langkah:", len(path) - 1)
