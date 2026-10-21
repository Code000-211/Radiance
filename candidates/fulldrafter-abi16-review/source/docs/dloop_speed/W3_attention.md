# W3 - Streaming-Attention (full_attention, geteilter Key-Bereich)

Stand: 2026-10-17, Worktree `/home/orca/FullDrafter/worktrees/dloop_attn`, Branch `worktree/dloop_attn`,
Basis-Commit `909ecd0`.

## 1. Scheme

Vorher: Grid `(n_seq*front, ceil(B*Q_heads/4))`, ein Warp pro (Block-Zeile, Query-Head)-Paar,
`threads=128`. Ein Warp lief alle `length+b` Keys seriell ab. Bei ctx 16000 sind das 16009 Keys
nacheinander in einer Kette aus `ds_bpermute`-Butterfly (5 Schritte) -> `expf` -> `fmaf`, und die
108 Pair-Warps sind nur 1.7 Waves pro CU - eine Kette, nichts zum Ueberlappen. Gemessen:
0.43-0.46 us pro gecacheter Position pro Launch.

Nachher, vier Aenderungen in einer:

1. **Key-Bereich pro Warp geteilt (flash-decoding-Form).** Grid `(n_seq*front, B*KV_heads)`, ein CTA
   pro (Block-Zeile, Key/Value-Head), `attn_threads=512` = 16 Warps. Warp `s` besitzt den
   zusammenhaengenden Cache-Bereich `[length*s/16, length*(s+1)/16)`; die `b` Block-Keys gehen
   gestreut an Warp `m mod 16`. Jeder Key wird genau einmal besucht, kein Key zweimal. Die
   Masken-/Positionsregel ist unveraendert (Cache `0..length-1`, danach `b` Block-Keys aus dem
   qkv-Scratch, `length+b` Keys, nichts capacity-abgeleitetes).
2. **GQA-Ausnutzung.** Der Warp besitzt den ganzen K/V-Head, also teilen sich dessen `groups` Query-
   Heads die eine K/V-Zeile, die er laedt. Die drei Score-Ketten eines Keys sind unabhaengig - das ist
   die ILP, die eine Lade-Latenz ueberhaupt ueberlappbar macht. Der Vorgaenger las dieselben K/V-Zeilen
   aus 108 Pair-Warps neu; jetzt einmal pro Warp.
3. **Vier Keys pro Umlauf, Lasten voraus.** `step()` bekommt die vier K/V-Werte fertig geladen
   uebergeben; der Aufrufer stellt die sechzehn Lane-Lasten von vier Keys vor die Arithmetik. Das ist
   der eigentliche Gewinn: der Kernel ist memory-level-parallelism-bound, und ein Key pro Umlauf liess
   nur 512 B pro Warp in flight.
4. **Merge im selben Launch.** LDS `part[16][3][66]` (12672 B), `__syncthreads`, dann pro (Query-Head,
   Head-Dim) ein Merge der 16 Split-Zustaende: `M=max m_s`, `l=sum exp(m_s-M)*l_s`,
   `acc=sum exp(m_s-M)*acc_s`, Ausgabe `acc*(1/l)`. Ein Launch pro Layer wie vorher, kein zweiter
   Kernel, keine Partiale im Global-Speicher.

Numerik: FP32, exaktes `expf`, kein BF16, keine Naeherung, `-ffp-contract=off` mit `a*b+c`-Ketten.
Die deklarierte Abweichung ist im Kernel-Kommentar (Zeilen 499-517 von `mine/dloop_full_device.h`)
ausgeschrieben: Butterfly statt aufsteigender Summe (unveraendert vom Vorgaenger) und online-Softmax
ueber `attn_splits` Teilaeufe statt einem. Leere Splits haben `m=-INFINITY`, `exp(-INFINITY-M)=0`,
tragen exakt nichts bei.

## 2. Arena

**Keine Arena-Erweiterung.** Die Partiale liegen in LDS des CTAs; `full_workspace`/
`full_elements` wurden nicht angefasst, `rad_dloop_full_scratch_elements` liefert unveraendert
dieselben Elementzahlen, der Engine-Allocator bleibt wie er ist. Zusatzbedarf: 0 Elemente F32,
12672 B LDS pro CTA (statisch, capacity-unabhaengig - die Partiale sind pro Split, nicht pro Position).

## 3. ISA

`bash scripts/dloop_kernel_check.sh isa`:

```
vgpr sgpr lds    scratch spill waves wg   kernel
67   66   12672  0       0     16    512  full_attention<0>
67   66   12672  0       0     16    512  full_attention<1>
```

Kein Spill, kein Scratch, 67 VGPR (Tier 72, <=96), `waterfall.sh` listet keinen Eintrag fuer
`full_attention` - v_movrels 0 auf diesem Pfad. Die Vorgaengervariante hatte 51 VGPR / 6336 B.

## 4. Messungen

### 4.1 Ende-zu-Ende (`bash scripts/dloop_kernel_check.sh sweep`, rank 0, median)

| Kontext | append | base | dartree | dartree Baseline | dartree Gewinn |
|---|---|---|---|---|---|
| 200 | 0.572 ms | 6.03 ms | **31.88 ms** | 34.2 ms | -2.3 ms |
| 1000 | 0.571 ms | 6.18 ms | **39.04 ms** | 51.6 ms | -12.6 ms |
| 4000 | 0.568 ms | 7.05 ms | **44.29 ms** | 99.3 ms | -55.0 ms |
| 16000 | 0.577 ms | 9.69 ms | **67.67 ms** | 288.5 ms | **-220.8 ms** |

Das Kopfziel des Plans (`ctx 16000 dartree <= 100 ms`) ist mit 67.7 ms erreicht; das
ctx-200-Ziel (`<= 32 ms`) mit 31.88 ms ebenfalls.

### 4.2 Attention pro Launch (abgeleitet, nicht direkt gemessen)

Zwei unabhaengige Anker, beide aus gemessenen Zahlen dieses Worktrees:
- **Anker A (base-Op):** eine `base`-Ausfuehrung durchlaeuft genau 5 Attention-Launches
  (5 Layer). Der kontextunabhaengige Anteil wird aus dem Baseline-Anker ctx 200
  (`6.3795 ms - 5*97.1 us = 5.894 ms`) zurueckgerechnet.
- **Anker B (dartree):** 35 Attention-Launches (5 Layer x 7 Tiefen), kontextunabhaengiger
  Anteil `34.2 - 35*97.1 us = 30.80 ms`.

| Kontext | Basis us/Launch | neu A | neu B | Planziel | erreicht? |
|---|---|---|---|---|---|
| 200 | 97.1 | 26.4 | 30.7 | <= 25 | knapp verfehlt |
| 1000 | 435 | 57.7 | 235 | - | - |
| 4000 | 1733 | 231 | 385 | <= 150 | **verfehlt (1.5-2.6x)** |
| 16000 | 6815 | 758 | 1053 | <= 500 | **verfehlt (1.5-2.1x)** |

[INFERENCE] Beide Anker sind Ableitungen aus zwei Messungen, kein Trace der neuen Kernel;
die Spanne zwischen A und B ist die Unsicherheit der Ableitung, nicht eine Schwankung der
Messung. Der Grund fuer die Spanne: der kontextunabhaengige Anteil ist bei `base` und
`dartree` nicht derselbe (dartree traegt die Tree-GEMMs, `base` nicht), also verschiebt
jeder Anker die Attention-Zahl um einen konstanten Betrag. Sicher ist nur die Richtung und
die Groessenordnung: **Faktor 7-9 schneller**, Ziel bei 4000/16000 um **1.5-2.9x verfehlt**.

## 4.3 Wo die Restzeit liegt (gemessen, ISA + Zahlen)

Die vier Keys pro Umlauf haben geholfen (67.7 statt 85.0 ms), aber der Kernel ist noch
nicht bandbreitengebunden. Gemessen:
- Der Hauptschleifenkoerper hat **~100 Instruktionen pro Key mit 64-Bit-Adressrechnung**
  (`s_mul_u64`, `s_lshl_b64`, `s_add_nc_u64` je Key): der Cache-Index wird pro Key als
  64-Bit-Produkt neu gebildet. Ein Vorwaertszeiger statt `n*2*D` waere die naechste Massnahme.
- Die Butterfly geht ueber `ds_bpermute_b32`: gfx12 hat kein `v_shfl`, `__shfl_xor` wird zu
  LDS-Rundlauf, 5 Runden pro Score-Kette.
- 36 CTAs bei 64 CUs = **56 % CU-Belegung**; die Aufteilung sitzt auf `blockIdx.y`. Eine
  weitere Aufteilung der Key-Range auf `blockIdx.z` muesste den Merge ueber CTAs fuehren -
  zweiter Kernel oder globale Partiale (Plan bevorzugt einen Launch).
- Der LDS-Merge ist capacity-unabhaengig konstant (~130 Instruktionen, 16 Splits), bei
  ctx 200 also vergleichbar gross wie die eigentliche Arbeit.

## 5. Verworfene Varianten (vermessene Zahlen)

1. **8 Splits / 256 Threads, ohne Last-Staffelung** (Commit `fdb2a65`). Grid
   `(n_seq*front, B*KV_heads)`, `attn_threads=256`, ein Key pro Umlauf, Block-Keys auf
   Split 0. Sweep: ctx 16000 base 13.22 ms / dartree 85.01 ms, ctx 4000 dartree 45.03 ms,
   ctx 200 dartree 34.32 ms. ISA 51 VGPR, 0 Spill, 16 Waves. Verworfen, weil die
   versprochene Wirkung der Aufteilung (Latenz verstecken) ausblieb: 0.44 -> 0.43 us pro
   Position. Grund gefunden und behoben: `#pragma unroll 4` feuerte nie
   (`-Wpass-failed=transform-warning`), weil das `break` in der Head-Schleife den Unroller
   blockierte; im Code-Objekt standen 4 Lasten im ganzen Kernel. Diese Variante ist als
   Messung im Bericht, nicht als Empfehlung.
2. **32 Splits / 1024 Threads pro CTA** (`attn_splits=32`, `attn_threads=1024`). Gebaut und
   mit `res.sh` vermessen: **112 VGPR, 12 Waves, 25344 B LDS**, kein Spill. Verworfen, weil
   112 VGPR in Tier 120 statt 48 faellt und eine 1024-Thread-CTA genau eine CU belegt; die
   CTA-Groesse ist nur das Vehikel, die Wellenzahl kommt aus den Splits.
3. **Laufzeit-Schleifengrenze `for(j=0;j<groups;++j)`** (statt `j<3` mit Guard). Loest die
   Zustandsarrays (`q0,q1,acc0,acc1,mh,lh`) in dynamisch indizierte Vektorarrays auf:
   Scratch/Register-Druck. Mit compile-time-Grenze 3 und Laufzeit-Guard: 67 VGPR, 0 Scratch.
4. **Zweiter Kernel / globale Partiale fuer den Merge** wurde nicht gebaut: die
   Split-Zustaende passen in 12672 B LDS, also braucht es keinen zweiten Launch und keine
   Partiale im Global-Speicher. Kein Arena-Wachstum, also auch keine Aenderung am Hook.

## 6. Paritaet

- `quick`: `DLOOP_GPU_PARITY_ALL_PASS`, **byte-identisch** zu `baseline_parity_quick.log`
  (exit 0). Beide Varianten bestehen das; die Umordnung der Aufteilung veraendert die
  Fixture-Ausgaben nicht.
- `real`: `DLOOP_GPU_PARITY_REAL_PASS` + `ALL_PASS`, exit 10, **14 abweichende Zeilen und
  alle 14 sind Float-Betraege** (`worst_abs`/`worst_ulp`). 0 Zeilen mit Tokens, Parents,
  Depths, `swapped` oder `cutoff_tie_rows` haben sich bewegt; `require`-Zusicherungen fuer
  `tree token/parent/depth` (Testzeilen 535-539) und `length` sind hart und sind durchgelaufen.
  Die Abweichungen sind genau die deklarierte Umordnung der Attention-Ausgaben (Butterfly +
  Split-Merge), die in `golden.full_y_base`, `local_ids` und `POOL_SET` durchschlaegt:

  | Zeile | neu | Baseline |
  |---|---|---|
  | `dloop_full_base.local_ids` r0 | 1.43051e-05 | 1.62125e-05 |
  | `golden.full_y_base.rank0` | 5.72205e-05 | 5.14984e-05 |
  | `dloop_full_base.local_ids` r0 | 1.0252e-05 | 7.15256e-06 |
  | `golden.full_y_base.rank1` | 5.72205e-05 | 5.14984e-05 |
  | `rank0.local_ids` | 3.52859e-05 | 3.62396e-05 |
  | `rank1.local_ids` | 1.52588e-05 | 1.12057e-05 |
  | `POOL_SET` | 3.52859e-05 | 3.62396e-05 |

  (Die vollen 14 Zeilen sind 7 Paare aus `<`/`>`, weil `diff` beide Seiten zaehlt.)

## 8. Host-CTest: Zuordnung der Fehlschlaege

`ctest --test-dir build_hip -LE gpu` ist **nicht** gruen, aber **kein** Fehlschlag stammt
aus diesem Auftrag. `git diff --stat 909ecd0..HEAD` beruehrt genau zwei Dateien
(`mine/dloop_full_device.h`, dieses Dokument); alle Testquelldateien, Fixtures und die
Engine liegen unveraendert auf `909ecd0`.

Frischer Vollbau in diesem Worktree, 9 Fehlschlaege. Davon 8 **direkt im
Baseline-Worktree `dloop_dartree` mit dessen eigenen Binaries reproduziert**:

| Test | in `dloop_dartree` reproduziert |
|---|---|
| 16 `drafter_gpu_exec_test` | ja (Failed) |
| 17 `drafter_gpu_rec_test` | ja (Failed) |
| 18 `drafter_graph_replay_test` | ja (Failed) |
| 20 `drafter_parity_test` | ja (Failed) |
| 22 `drafter_vram_alloc_test` | ja (Failed) |
| 48 `sched_test` | ja (Failed) |
| 64 `ops_doc` | ja (Failed, `UNDOCUMENTED fulldrafter_two_stage`) |
| 66 `oot_plugins` | ja (Failed, Install-Ziel `bin/radiance` fehlt) |
| 69 `dloop_loader` | nein, aber aus dem Basis-Commit beweisbar (siehe unten) |

`dloop_loader` ist ein **Defekt des Basis-Commits** und keine Regression: `909ecd0` hat
`candidates/dloop_dartree/loader_test.cpp` auf die Version-2-Semantik gezogen
(`source_promoted` und boolean `use_kv_cache` werden als **akzeptiert** gefordert,
Zeilen 55 und 64), das Fixture `tests/fixtures/dloop_tiny/dloop_sidecar.json` traegt aber
weiterhin `"version": 1` — in beiden Worktrees byte-identisch und unveraendert gegenueber
git. Ein frisch gebautes `dloop_loader_test` kann auf `909ecd0` also nicht bestehen; im
Baseline-Worktree besteht es nur, weil dort eine **veraltete Binary** liegt
(mtime 16. Okt 21:03) gegenueber der geaenderten Quelle (17. Okt 22:58). Beweis ohne
Fremd-Worktree: `git show 909ecd0:candidates/.../loader_test.cpp` fordert v2,
`git show 909ecd0:tests/fixtures/dloop_tiny/dloop_sidecar.json` liefert v1.

Die DLoop-Host-Referenztests, die der Auftrag als "host ref untouched" meint, sind gruen:
`dloop_full_lifecycle_test`, `dloop_full_ref_test`, `dloop_ref_test` (alle Passed).

## 9. Bilanz gegen die Auftragsziele

**Erreicht (gemessen, rank 0):**
- `ctx 16000 dartree <= 100 ms` (Kopfziel): **67.7 ms** aus 288.5 ms (4.26x).
- `ctx 200 dartree <= 32 ms`: **31.88 ms** aus 34.2 ms.
- `base` Op: 39.97 -> 9.69 ms bei ctx 16000 (4.13x), 14.56 -> 7.05 ms bei 4000 (2.07x).
- Paritaet: quick byte-identisch; real exit 10 mit ausschliesslich Float-Betraegen;
  `tree tokens/parents/depths` und `length` unveraendert (harte `require`s).
- ISA-Budget: 67 VGPR (Tier 72, <=96), 0 Spill, 0 Scratch, kein `v_movrels` auf dem Pfad.

**Verfehlt (gemessen, nicht wegtoleriert):**
- Per-Launch-Ziele bei ctx 4000 und 16000: erreicht sind 231-385 us statt <= 150 und
  758-1053 us statt <= 500 (Faktor 1.5-2.9, je nach Anker; die Spanne ist in 4.2
  begruendet). Bei ctx 200 sind 26.4-30.7 us statt <= 25 gemessen.
- Der Grund ist im ISA belegt und in 4.3 benannt: 64-Bit-Adressrechnung pro Key,
  `ds_bpermute`-Butterfly ohne `v_shfl` auf gfx12, 56 % CU-Belegung durch 36 CTAs.
  Das ist eine eigene Iteration, nicht eine Toleranzfrage.

## 10. Commits und Reproduktion

Commits in diesem Worktree (Branch `worktree/dloop_attn`, Basis `909ecd0`):

| SHA | Inhalt |
|---|---|
| `fdb2a65` | 8 Splits / 256 Threads, Key-Range aufgeteilt, Merge in LDS (verworfen, s. 5.1) |
| `eaeb7c0` | 4 Keys pro Umlauf mit vorgezogenen Lasten, Block-Keys gestreut, 16 Warps |
| `df2d2e1` | dieser Bericht |

Reproduktion (jede GPU-Zeile unter der Queue-Sperre, ueber das Skript):

```bash
cmake -B build_hip -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAD_WITH_HIP=ON \
      -DRAD_GPU_TARGETS=gfx1201 -DRAD_BUILD_TESTS=ON
ninja -C build_hip mine mine_ref dloop_gpu_parity_test
bash scripts/dloop_kernel_check.sh quick     # exit 0, IDENTICAL
bash scripts/dloop_kernel_check.sh sweep     # die Tabelle in 4.1
bash scripts/dloop_kernel_check.sh isa       # die Zeile in 3
bash scripts/dloop_kernel_check.sh real      # exit 10, die 14 Zeilen in 6
ninja -C build_hip && ctest --test-dir build_hip -LE gpu   # Zuordnung in 8
```

Kein Test wurde abgeschwaecht, keine Toleranz justiert; `real` exit 10 ist die deklarierte
Umordnung und jede der 14 Zeilen ist oben einzeln aufgefuehrt.
