# DLoop Speed — Welle 1 Ergebnis und Welle 2 Plan

**Stand:** 2026-10-18, nach Integration von W1–W4 in `worktree/dloop_dartree`
(Commits bis `495bf44`). Baseline-Commit der Messreihe: `909ecd0`.

## 1. Welle 1: gemessen (immer rank 0, Median, gleiche Methode wie davor)

**Decode-Bench der drei Ops** (`scripts/dloop_kernel_check.sh bench CTX`):

| Kontext | append | base | dartree | dartree vorher |
|---|---|---|---|---|
| 200 | 0.51 ms | **3.65 ms** | **8.14 ms** | 34.2 ms |
| 1000 | 0.51 ms | **3.85 ms** | **10.72 ms** | 51.6 ms |
| 4000 | 0.50 ms | **4.60 ms** | **17.31 ms** | 99.3 ms |
| 16000 | 0.51 ms | **7.28 ms** | **45.48 ms** | 288.5 ms |

**Engine, BetterBench single-stream decode** (c16k-Sidecar, TP2 exact, Budget 3,
adaptiv aus, je Kategorie 1 Lauf — Rauschen beachten):

| Kategorie | alt (51248ae, hr384) | DLoop vor W1 | DLoop Welle 1 |
|---|---|---|---|
| chat | 92.6 | 23.7 | 68.2 |
| code | 167.8 | 35.6 | 92.3 |
| file_edit | 169.5 | 50.5 | 110.4 |
| json | 168.5 | 39.8 | 103.5 |
| math | 197.5 | 42.1 | 107.3 |
| prose | 97.2 | 23.4 | 67.2 |
| reasoning | 127.4 | 30.1 | 87.1 |
| summarization | 103.5 | 24.2 | 67.8 |
| **weighted** | **~131** | **~35** | **88.5** |

**Welle 1 ist ein Fortschritt, NICHT das Ziel.** Die Konvergenzbedingung
(alter Drafter je Kategorie) ist verfehlt: code 92 vs 168, math 107 vs 197.
Kein Anspruch „auf dem Niveau" — die Tabelle steht hier, damit niemand ihn liest.

## 2. Wo die restliche Zeit steckt (Phasen-Timing, 24 Steps des Bench-Laufs)

| Phase | Median | Anteil |
|---|---|---|
| collect (Verify) | 14.6 ms | 63 % |
| draft | 7.6 ms | 33 % |
| ranks | 0.4 ms | 2 % |
| commit | 0.4 ms | 2 % |

Draft ist von 40 ms auf 7.6 ms — Welle 1 hat ihr Soll erfüllt (8–12 ms erwartet).
**Der Flaschenhals ist jetzt `collect`**, nicht der Draft. Zwei Fragen sind dort offen:

1. **Budget-3-Restriktion.** Der alte Drafter lief mit Budget 15 (tiefe Fenster);
   §4 verbietet statisch >3 nur OHNE adaptiven Fallback. `NUM_SPECULATIVE_TOKENS=3`
   künstelt tok/update (~2.0 vs ~3.4 beim alten Lauf) und damit tok/s. Erwartung:
   ein legitimer 15er-Fenster-Betrieb (adaptiv an ODER dokumentierter Fallback)
   hebt tok/update und damit alle Kategorien.
2. **Verify-Kosten.** `collect` 14.6 ms bei n_seq=1 und 3–4 Spec-Tokens. Ein neuer
   --profile-ops-Lauf (unprofiliert nicht vergleichbar!) muss zeigen, was darin
   trunk-Kernel gegen Verify-Baum-Kernel sind, bevor Welle-2-Scheiben vergeben werden.

## 3. Welle 2 — nächste Schritte (in dieser Reihenfolge)

1. **P146**: Adaptives/spec-Fenster für DLoop klären und mit Budget 15 messen
   (dloop_testrun.sh erlaubt >3 nur mit Fallback; dokumentieren, nicht umgehen).
   Messung: BetterBench wie oben, Aufteilung draft/collect im Log.
2. **P147**: Neuen --profile-ops-Lauf des INTEGRIERTEN Stands aufnehmen und
   `collect` aufschlüsseln (unprofilierte Phasen sind die Wahrheit, der alte
   profilierte Lauf ist nicht vergleichbar). Danach erst: Verify-Kernel-Scheiben.
3. **P148** (optional, nach 2): Fusion der 4.5-µs-Dispatch-Lücken im Draft, wenn
   der neue Trace Lücken >10 % zeigt (Kandidaten stehen in DLOOP_SPEED_PLAN.md §11).

## 4. Parität nach Welle 1 (Nachweis)

- `quick`: byte-identisch zur Baseline (149 Zeilen).
- `real`: exit 10, 14 Zeilen Float-`worst_abs` der Base-Ausgaben durch die
  deklarierte Attention-Umordnung (`"numerics": "reordered"` in
  `candidates/dloop_dartree/gate.json`); Tokens/Parents/Bäume unverändert,
  `swapped=0`, `cutoff_tie_rows=0`.
- ISA: 0 Spills, 0 v_movrels in den DLoop-Kernel, skinny 32–57 VGPR,
  attention 67 VGPR, merge 41 VGPR.
- `dloop_loader`-Fix: Test liest die v2-Fixture (`tests/CMakeLists.txt`),
  3/3 grün. Sched-Test-Fälle bleiben rot (Altlast aus 51248ae, nicht unser Pfad).

## 5. Ein offener Konflikt fürs Review

`two_stage_draft_depth_policy`/`the_draft_depth_is_stable...` in
`tests/sched_test.cpp` prüfen Block-Drafter-Erwartungen (7/3), die
`tree_state`-Realität (15/4) nicht mehr abbilden — der Test ist älter als der
tree_state-Zweig. Er gehört neu geschrieben, nicht wegretuschiert; das ist ein
eigenes Ticket, kein Teil der Kernel-Arbeit.
## 6. Budget-Messung mit echtem Fallback (2026-10-18, build wave1-fallback)

| Konfiguration | combined tok/s | update p50 | tok/update | draft ms | collect ms |
|---|---|---|---|---|---|
| Budget 3, adaptiv aus (statisch, §4-ok) | **88.5** | ~23 ms | 1.5–2.6 | 7.6 | 14.6 |
| Budget 15, adaptiv an (Fallback aktiv) | **55.7** | ~30 ms | 1.4–2.3 | 16.4 | 13.5 |

**Befund:** Das tiefe Fenster verdoppelt die Draft-Kosten und kauft bei diesem
Checkpoint (step 52500, fixed variant) keine Annahme: tok/update sinkt leicht,
collect bleibt gleich. Das Geräte-Budget (Seitenwagen `budget=15`) steuert den
Baumlauf direkt und ist eine Declare-Eigenschaft; der EMA-Fallback im Scheduler
kann nur die verifizierten Zeilen kürzen, nicht den Baumlauf verbilligen.
**Messdaten:** `betterbench_integrated_20261018_012529.json` (b3) und
`betterbench_integrated_b15_20261018_015117.json` (b15), Phasen aus den
`engine_wave1_*.log`-Step-Zeilen.

**Konsequenz, korrigiert 2026-10-18:** Budget 3 ist die §4-konforme
Konfiguration, und der einzige sauber gemessene Punkt. Der Budget-15-Lauf dieser
Tabelle lief mit dem inzwischen wegen Pfad/Knoten-Einheitenfehlern REVERTIERTEN
Fallback — er beweist NICHT, dass der Checkpoint tiefe Fenster nicht konvertiert;
er zeigt nur, was eine kaputte Politik misst. Der Vergleich 3 gegen 15 steht aus,
bis das Fenster-Vertragsdesign steht. §4 bleibt erfüllt: statisch 3.
