# P150 — hc_read-Messung und Entscheidung (Trunk-Hebel)

**Stand:** 2026-10-18. Messbasis: `reports/dloop_engine/engine_profile_ops_wave1.log`
(letzte Profiltabelle, integrierter Welle-1-Stand, Budget 3, 2625 Decode-Schritte).

## Gemessen

| Op | Aufrufe | gesamt ms | µs/Aufruf | ms/Schritt |
|---|---|---|---|---|
| hc_read | 254 625 | 8 666.6 | 34.0 | **3.30** |
| ar_gather_hc_write | 126 000 | 3 130.8 | 24.9 | 1.19 |
| ar_hc_write | 126 000 | 2 240.8 | 17.8 | 0.85 |

Geometrie (aus der Profiltabelle): `n=2560 hc=4 lowrank=320 eps=1e-06 dtype=bf16
wadd=1 inject=1 group=128 rotate=0/1 mix=e4m3`, gewählter Kernel: libr4d
(`hc_read_e4m3`-Reihe, `libr4d/r4d_selftest.cpp:5094` referenziert sie).

~97 hc_read-Aufrufe pro Schritt (12 Blöcke × 2 Lese + MTP/Reste) à 34 µs bei
einer Decode-Zeilenzahl von nur M ≈ 3–5. **Erwartung (nicht Messung):** ein
Kernel, der auf Prefill-M (bis 2048) getilt ist, verschwendet auf M<8 fast
seine gesamte Parallelität; ein Decode-Spezialpfad (kleine M, gleiche
Arithmetik, `libref/ref_hc.cpp` als Orakel) sollte bei 34 µs auf einstellige
µs fallen und damit ~2.5–3 ms/Schritt von der Trunk-Steuer abziehen — für
beide Drafter gleichermaßen.

## Entscheidung

1. **Kein drive-by in diesem Checkout.** hc_read ist Trunk (libr4d), nicht
   DLoop. AGENTS.md §5: Hebel gegen den Trunk laufen als eigener Hebel durch
   Queue/Gate, nicht nebenbei im DLoop-Worktree.
2. **Empfehlung an die Lever-Queue:** Hebel-Idee „hc_read-Decodepfad (kleine M)"
   mit genau dieser Tabelle als Messbasis und `libref/ref_hc.cpp:1-18` als
   Orakelvertrag (die Numerik-Diskussion steht dort: drei Divisionen, Write
   liest das UNNORMIERTE h). `tier:heavy` (neuer Kernel, spekulativer
   Decoding-Pfad berührt).
3. P148 bleibt offen bis ein Dispatch-Trace des integrierten Budget-3-Stands
   die Lückenfrage beantwortet; diese Messung hier ist Op-Zeit, keine
   Lückenmessung.

**Dieser Bericht ersetzt keine Lever-Planung; er liefert die Messbasis dafür.**