## Aktueller Nachtrag: feste 8 Positionen / 15 Knoten

Die folgenden älteren Budget-3-Betriebsregeln sind für diesen Drafter aufgehoben.
Aktuelle Messkonfiguration: `RADIANCE_DARTREE_VARIANT=fixed`,
`RADIANCE_DARTREE_BUDGET=15`, `NUM_SPECULATIVE_TOKENS=15`,
`RADIANCE_ADAPTIVE_SPEC=0`. Acht Vorschlagspositionen sind nicht dasselbe
wie fünfzehn Baumknoten.

Integriert: Dyn-GEMM/Conv-Pre-Fusion, wiederverwendete RoPE-Koeffizienten,
RoPE im Attention-Ladepfad, Skinny-K-Schleifen mit Unroll 4,
LM-Head-Decode-Mapping mit BF16-LDS-Gewichten sowie zeilenweise CTAs
für Initial-Norm und Finish/Norm. Akkumulation und Zwischenformate bleiben
unverändert; Append bleibt unverändert.

GPU-Einzel-Op-Messung, reale L5-Gewichte, `n_seq=1`, logischer Rank 0,
GPU 0, feste 15 Knoten; keine TP2-Engine-Durchsatzmessung:

| Op / Kontext | Vorher (ms) | Integriert (ms) |
|---|---:|---:|
| Base / 200 | 3.6511 | 2.4971 |
| DARTree / 200 | 8.1481 | 5.6839 |
| Base / 16000 | 7.2820 | 4.7076 |
| DARTree / 16000 | 45.4703 | 26.3457 |

Artefakte unter `reports/dloop_engine/`:
`fusion_original_decode_xvZMpXHL` und `integrated_x1UgP3v1`.
Der integrierte Lauf meldet `DLOOP_GPU_PARITY_ALL_PASS`.
Die aufgezeichneten Geräteausgaben stimmen mit dem Ausgangsplugin bytegenau
überein, einschließlich Top-k-IDs und Reihenfolge:

- vollständige Parität: `3313a5b80d9c684c636f260273c4f0079053532512d18eb1767ffb926d646a3b`
- Decode Kontext 200: `5ae8f7b1acd17b2ae0c10e32d6dbf586c4c66f5a6ee224973eb788ec66744524`
- Decode Kontext 16000: `4081e27799d5bd3710a062acb77b3ad4cb97678d5fe00934322289e94a64b8d4`

Trace vor den abschließenden Norm-CTA-Änderungen:
`launch_analysis_bzKl4hNM` → `launch_analysis_SENSZeTo`;
Base 66 → 51 Launches, DARTree 467 → 362, Append weiterhin 17.

Verworfen: Head-Streaming, gemeinsame Attention-Zeilen und Skinny-Spalten-CTAs
(langsamer; Quellen und Postmortems unter `attic/2026-10-18_*`).
Ebenso verworfen: größere Head-Kacheln, Unroll 8 und 32 Attention-Splits.
Die 32-Split-Variante änderte zusätzlich die Ausgabebytes.

CPU-Referenztests des aktuellen Host-Builds: `dloop_full_ref_test`
(17 Fälle, 31367 Checks) und `dloop_ref_test` (6 Fälle, 172071 Checks),
jeweils Exit 0. Diese Tests prüfen die Host-Referenz, nicht die GPU-Optimierung.
Der HIP-Build meldet für `mine` und `dloop_gpu_parity_test` keine offenen Buildschritte.
Native LSP-Diagnostik benötigt die HIP-Compiler-Datenbank und HIP-Modus für
die Device-Header; die ursprünglichen Include-Fehler waren Konfigurationsfehler.

### TP2-Engine-A/B, feste 15 Knoten, adaptiv aus

Ausgangsplugin: `fixed15_current_iTYH6fVY`.
Integriertes Plugin: `fixed15_current_cKLTMhxb`.
Beide Läufe endeten mit `RUN_EXIT=0 STOP_EXIT=0 COPY_EXIT=0`.
Identisches Engine-Binary, Plugin-Hashes in den jeweiligen `build.sha256`.
Die zwischenzeitlichen Engine-Läufe mit `adaptive=1` zählen nicht zu diesem Vergleich.

| Messgröße | Vorher | Integriert |
|---|---:|---:|
| Draft-Phase, Median der 128-Schritt-Mittelwerte | 17.255 ms | 13.190 ms |
| Gesamtschritt, gleicher Aggregationstyp | 41.600 ms | 37.540 ms |
| Collect-Phase, gleicher Aggregationstyp | 23.055 ms | 23.290 ms |
| Aufgezeichnete Dispatches im Draft-Pass 1 | 543 | 423 |
| BetterBench gewichteter Decode-Wert | 55.1 tok/s | 63.6 tok/s |

Phasenbasis: 20 bzw. 21 protokollierte 128-Schritt-Blöcke, einschließlich Warmup.
BetterBench: Temperatur 0.7, acht Kategorien, je ein Warmup und ein Messlauf.
Die Durchsatzwerte sind eine Einzelmessung mit stochastischer Ausgabe,
kein präziser isolierter Speedup und kein Annahmeraten-Gate.
Die Phasenmediane lassen sich nicht addieren.
Kein Gate-PASS und keine Freigabe für eine Übernahme nach main.

Zusätzliche, strengere Decode-Orakelprobe:
`RADIANCE_DLOOP_DECODE_BENCH=200 RADIANCE_DLOOP_DECODE_VERIFY=1`
prüft die synthetischen Single-Sequence-Eingaben gegen die Host-Referenz.
Sie scheitert **auch mit dem unveränderten Ausgangsplugin** an
`dloop_full_base.y_base[419]`: Gerät `-0.010086`, Referenz `-0.010125`.
Beide Läufe enden mit Exit 1; die Toleranz bleibt unverändert.
Belege: `decode_oracle_HPXjFyyZ` (Ausgangsplugin) und
`decode_oracle_wrPsnhCp` (integriert). Diese Probe ist kein PASS.
Die bytegenaue Geräte-A/B-Prüfung und die vollständige reale Golden-Prüfung
sind davon getrennte Nachweise. Die Ursache der bestehenden Host/GPU-Abweichung
ist hier nicht geklärt.

Der abschließende Neubau (`final_build.log`, Exit 0) erzeugt denselben
Plugin-SHA256 wie der gemessene integrierte Stand:
`33b4205beea7ca4eb7e04029a4b1d25793fce1f90862dffd9c14cd95a4a673f1`.

# DLoop/DFlash2-Drafter — Stand nach Welle 1 + P149 (2026-10-18)

**Ziel:** mindestens so schnell wie der alte 51248ae-Drafter. **Status: nicht erreicht.**
Der Stand ist ehrlich dokumentiert; die verbleibende Differenz ist gemessen und
zugeschrieben.

## Gemessen (immer gleiche Methode, rank 0, Budget 3, adaptiv aus)

| Größe | vorher | jetzt | Faktor |
|---|---|---|---|
| dartree-Op ctx 200 | 34.2 ms | **8.17 ms** | 4.2× |
| dartree-Op ctx 16000 | 288.5 ms | **45.5 ms** | 6.3× |
| base-Op ctx 200 | 6.36 ms | **3.80 ms** | 1.7× |
| Engine decode combined | 35.5 tok/s | **88.5 tok/s** | 2.5× |
| gdn_conv_tree (Engine-Geometrie) | 9.3 µs | **7.0 µs** | 1.3× |

Alter Drafter je Kategorie (hr384): chat 92.6 / code 167.8 / math 197.5 tok/s.
Jetzt: chat 68.2 / code 92.3 / math 107.3 tok/s. **Jede Kategorie bleibt langsamer.**

## Warum (aus dem integrierten Profil, PROFILE_WAVE1.md)

- Trunk-Steuer ≈ 16 ms/Schritt (GEMM 10.4 + hc_read/hc_write 5.4) — für jeden
  Drafter gleich.
- DLoop-Eigenanteil: draft 6.3 + gdn ≈ 1.3 ms — Welle 1 + P149 haben diesen
  Teil von ~26 ms auf ~8 ms gedrückt.
- **Die Annahmelänge ist der Hauptverdächtige** (1.5–2.6 vs 3.4 tok/update),
  aber das ist [INFERENCE], kein bewiesener Befund: der Budget-15-Lauf (55.7
  tok/s) lief mit dem inzwischen wegen Einheitenfehlern REVERTIERTEN Fallback
  und beweist nichts über den Checkpoint. Ein gültiges >3-Fenster-Experiment
  braucht erst das Pfad/Knoten-Einheiten-Design (offen).
- **P148 (gemessen, verworfen):** Der Budget-3-Dispatch-Trace zeigt 570 µs
  Lücken pro Baumaufruf (21 % von 2.78 ms Wandzeit, Median-Lücke 4.2 µs) —
  die Fusion-Deckel-Obergrenze ist ~2.4 % der Schrittzeit. P148_fusion.md.
- **P150 (gemessen, Entscheidung gefällt):** `hc_read` kostet 3.3 ms/Schritt
  (34 µs/Aufruf bei Decode-M≈3–5, 97 Aufrufe/Schritt), Kern in libr4d.
  Trunk-Hebel → Hebel-Idee „hc_read-Decodepfad“ (Messbasis und Vertrag:
  P150_hc_read.md; Entwicklungsroute: eigener Kernel im mine-Plugin, gewählt
  über die Hierarchie, validiert gegen libref — nicht umgangen).

## Parität (Nachweis, nicht Behauptung)

- `quick` byte-identisch zur Baseline (149 Zeilen) nach jedem Schritt.
- `real`: 14 deklarierte Float-Zeilen (Attention-Umordnung,
  `"numerics": "reordered"`), `swapped=0`, Bäume/IDs unverändert.
- GDN-Ablehnungspfad: 6/6 REFUSAL_PASS auf der GPU (neue `run_refused`-Methode
  im Harness); Bereichsschutz vor jedem Parent-Laden ist erprobt.
- ISA: 0 Spills, 0 neue Waterfalls, VGPR-Budget überall eingehalten.

## Offene Punkte

1. **P150 (offen, nächster Schritt):** `hc_read` ist gemessen (3.3 ms/Schritt,
   34 µs/Aufruf bei Decode-M≈3–5, 97 Aufrufe/Schritt, Kern in libr4d). Route:
   ein Decode-spezialisierter hc_read-Kernel IM MINE-PLUGIN, über die
   Plugin-Hierarchie für diese Geometrie gewählt, gegen librefs Orakel
   validiert, dann durchs Gate. Messbasis und Vertrag: P150_hc_read.md.
   **Dieser Hebel ist noch nicht gebaut** — er ist der größte offene Posten.
2. **Sched-Altlasten und Sidecar/Kapazität:** als Tickets #1 und #2 im
   Tracker geführt (audit:bug, needs-triage).
3. **Budget>3-Fallback:** der Baumfenster-Fallback (Pfad- gegen Knoten-Einheiten)
   wurde revertiert, bis ein sauberes Vertragsdesign steht; Budget 3 ist die
   §4-konforme Konfiguration dieses Checkpoints (der 3-vs-15-Vergleich ist
   wegen des revertierten Fallbacks offen).