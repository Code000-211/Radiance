# P148 — Dispatch-Lücken des integrierten Budget-3-Draft-Pfads (Messung)

**Methode:** `RADIANCE_DLOOP_BENCH_BUDGET=3` (neu, validiert) + rocprofv3-Kernel-Trace
des Decode-Benchs, dartree-Aufruf segmentationiert zwischen `tree_initialize` und
`tree_write_tree`. 32 Aufrufe, alle 137 Kernel.

**Gemessen (Median über 32 Aufrufe):**

| Größe | Wert |
|---|---|
| Kernel/Aufruf | 137 |
| Wandzeit/Aufruf | 2 782 µs |
| Busy-Summe | 2 185 µs |
| Lücken gesamt | 570 µs (**21 % der Wandzeit**) |
| Median-Lücke | 4.2 µs |

**Fusion-Deckel (Messung, keine Schätzung mehr):** volle Beseitigung jeder Lücke
spart maximal 570 µs pro Baumaufruf. Der Baumaufruf läuft einmal pro Decodestep,
der unprofilierte Schritt liegt bei ~23–24 ms (Phasenmessung). Also **maximal
~2.4 % Schrittzeit** — für die EINE Fusion, die alles eliminiert, was es nicht
gibt. Einzelne Fusionen (z. B. rope in QKV-Epilog, conv_pre in Prologue)
bringen je einen Bruchteil davon.

**Entscheidung:** P148 bleibt verworfen — jetzt mit Messung statt Schätzung:
die Deckel-Obergrenze von 570 µs/Aufruf rechtfertigt keine Fusionstage gegen
ein 8.3-ms-Ziel, das Welle 1 bereits um 76 % unterboten hat. Der nächste
wirkliche Hebel bleibt P150 (hc_read, 3.3 ms/Schritt) und danach Training.

**Daten:** `reports/dloop_engine/trace_decode_bench_wave1/`.
