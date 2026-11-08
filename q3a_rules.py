# Q3a - Knowledge Representation (aturan IF-THEN) + Reasoning (Forward Chaining)
rules = [
    ({"demam", "batuk", "sesak_napas"},          "infeksi_pernapasan"),
    ({"demam", "nyeri_otot", "menggigil"},       "influenza"),
    ({"demam", "ruam", "nyeri_sendi"},           "dengue_suspect"),
    ({"infeksi_pernapasan", "sesak_napas"},      "rujuk_dokter_paru"),
    ({"dengue_suspect", "mual"},                 "tes_darah_segera"),
]

def forward_chaining(facts):
    facts, trace, changed = set(facts), [], True
    while changed:
        changed = False
        for premises, conclusion in rules:
            if premises <= facts and conclusion not in facts:
                facts.add(conclusion); changed = True
                trace.append((sorted(premises), conclusion))
    return facts, trace

patient = {"demam", "batuk", "sesak_napas"}
facts, trace = forward_chaining(patient)
print("Fakta awal :", sorted(patient))
for p, c in trace:
    print(f"IF {p} THEN {c}")
print("Kesimpulan :", sorted(facts - patient))
