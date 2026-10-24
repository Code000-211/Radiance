# P149 — GDN-Baum-Verify-Kernel: integriertes Ergebnis

**Arbeiter:** W5GdnKernels (Branch `worktree/dloop_gdn2`, Commit `dbeb4fd`),
integriert als `fad042c` in `worktree/dloop_dartree` (Hilfsskripte des Arbeiters
nicht übernommen). Parität danach selbst nachgemessen: `quick` IDENTICAL,
`bench 200` dartree 8.17 ms / base 3.81 ms, `dloop_*`-Host-Tests 4/4 grün.

## Attribution (gemessen, Engine-Geometrie nseq=4 nnode=2)

| Kernel | µs/Aufruf | Blöcke | Diagnose |
|---|---|---|---|
| gdn_conv_tree | 9.3 | 160 | Prolog-/Launch-nah |
| gdn_recurrent_tree | 17.1 | 96 | Prolog-/Launch-nah |
| gdn_tree_commit | 13.5–17.9 | 136 | Prolog-/Launch-nah |

Leerer Kernel gleicher Form: 2.7–2.9 µs; mit Prolog-Barriere 3.2–3.4 µs.
Der Launch-/Prolog-Boden dominiert, nicht die Knotenschleife — das Profil
hat das vorhergesagt.

## Ergebnis

**Landevorteil:** `valid()` und `tree_paths()` zu `check_and_paths()`
vereinigt (drei Barrieren → zwei).

| Kernel | vorher | nachher |
|---|---|---|
| gdn_conv_tree | 9.4 µs | **7.0 µs** (Ziel ≤ 8 erreicht) |
| gdn_tree_commit (nn=9/16) | 13.5 / 17.4 µs | 11.3 / 15.1 µs |
| gdn_recurrent_tree | 17.1 µs | 17.1 µs (am Boden) |

`quick` byte-identisch (149 Zeilen), VGPR 51/182/71 unverändert, 0 Spill,
`v_movrel` 3 in conv_kernel (vorbestehend, ebenso in der Baseline).

## Verworfen, mit Zahlen

1. q/k nach LDS stagen: nach Bit-Fix IDENTICAL, aber **langsamer**
   (122.9 → 134.5 µs bei 96 Blöcken; LDS 16 916 → 25 108 B kostet residente
   Blöcke).
2. `__syncthreads_or` in commit: kernarg 672 → 928 B über dem 768-B-Slot,
   Start verweigert. Rückabgewickelt.

## Konsequenz

`gdn_conv_tree` ist um 25 % schneller und am Ziel; `recurrent` und `commit`
sitzen am Launch-Boden (2.7–3.4 µs leer) — weitere Fusionen innerhalb der
Kernel-Verträge sind ausgereizt. Der verbleibende gdn-Anteil (~1.3 ms/Schritt)
ist Dispatch-Struktur, nicht Kernel-Arithmetik. Das Profil-Dokument behält
recht: der nächste echte Hebel ist Trunk-seitig (P150: `hc_read`-Kernel messen),
danach Training.
## Nach der Integration nachgemessen (Orchestrator)

- `quick`: IDENTICAL (149 Zeilen).
- `real`: exit 10 mit denselben 14 deklarierten Welle-1-Float-Zeilen (Attention-Umordnung), keine neuen.
- Ablehnungssonden (neu im Harness, `run_refused`): 6/6 REFUSAL_PASS — die Geräteker
  starten bei fehlgeschlagener Metadatenprüfung (parent 999, cu 32 Knoten, bos=-1),
  verweigern ohne Schreibzugriff und ohne Fault. Der Bereichsschutz im Prolog
  (bos/nnode/bos+nnode gegen c.tokens vor jedem Parent-Laden) ist damit auf der
  GPU erprobt, nicht nur behauptet.
- `bench 200`: dartree 8.17 ms, base 3.80 ms (dartree unverändert, wie erwartet: gdn
  läuft im Verify-Pfad, nicht im Draft-Op).
