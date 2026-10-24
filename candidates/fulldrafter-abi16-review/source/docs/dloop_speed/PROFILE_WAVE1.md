# DLoop Welle 1 — Profil des integrierten Stands (collect aufgeschlüsselt)

**Stand:** 2026-10-18, Build = Welle-1-Kernel (Commit-Kette bis `1f0c10f`), Budget 3,
adaptiv aus, c16k-Sidecar, TP2 exact, `--profile-ops` (diagnostischer Lauf, nicht
messbar-komparabel, siehe Spaltenkopf des Profils).

## Ablaufprofil (20 Step-Zeilen, Median)

| Phase | Median | Anteil |
|---|---|---|
| ranks | 23.8 ms | **71 %** |
| collect | 0.1 ms | 0 % |
| commit | 2.5 ms | 8 % |
| draft | 6.8 ms | 20 % |
| step | 33.3 ms | |

Unter `--profile-ops` wandert die Trunk-Verarbeitung in die `ranks`-Phase (die
Kernel-Zeit wird synchron abgerechnet); die unprofilierten Läufe zeigen dieselbe
Arbeit als `collect` 14.6 ms (b3) bzw. 13.5 ms (b15). Die Aufschlüsselung unten
gilt für beide.

## Pro-Op-Gerätezeit, 2625 Decode-Schritte (letzte Tabelle des Profils)

| Gruppe | gesamt ms | ms/Schritt |
|---|---|---|
| Trunk-GEMM (moe_gemm_q, gemm_nt_q, gemm_nt) | 27 403 | **10.44** |
| HC-Lese/Schreib-Verkehr (hc_read, ar_gather_hc_write, ar_hc_write, hc_enter) | 14 060 | **5.36** |
| DLoop-Draft-Ops (base, dartree, append, pool) | 16 623 | **6.33** |
| Router/Quant (router_topk_scatter, quant-ops) | 3 889 | 1.48 |
| GDN-Baum-Verify (gdn_recurrent_tree, commit, conv) | 4 137 | 1.58 |
| Logits | 1 423 | 0.54 |
| Attention (attn_paged, qsa_*) | 1 787 | 0.68 |

## Lesart

1. **Der Trunk, nicht der Drafter, begrenzt jetzt den Schritt.** Draft-Ops kosten
   6.3 ms/Schritt (Welle 1 hat hier 2.5–3× gespart); Trunk-GEMM + HC-Verkehr
   kosten 15.8 ms/Schritt. Die Verify-Last ist der Trunk-Forward über die
   angenommenen Zeilen — der alte Drafter zahlt dieselbe Trunk-Steuer.
2. **`hc_read` 8.7 s / 254 625 Aufrufe ist KEIN reiner Speicherverkehr.** Die
   Op ist der GATED RESIDUAL des Modells (libref/ref_hc.cpp:1-18,
   docs/OPS.md:183): gruppierte RMSNorm, zwei Low-Rank-GEMMs, Sigmoid/SiLU-
   Mixung des HC-Stroms — Rechen- UND Speicherlast. Die 3.3 ms/Schritt sind
   nicht per se unveränderlich; dieselben Fragen wie bei den GEMMs (Auslastung,
   Fusion) gelten. Derselbe Satz läuft für den alten Drafter identisch — ein
   Trunk-Hebel, kein DLoop-Hebel.
3. **DLoop-eigene Restposten:** `gdn_*` 1.6 ms/Schritt (Verify-Baum des Drafters)
   und in `draft` selbst 6.3 ms. Welle-2-Kandidaten mit echtem DLoop-Hebel:
   die `gdn_recurrent_tree`/`gdn_tree_commit`/`gdn_conv_tree`-Kernel
   (94 284 Aufrufe, zusammen 4.1 s).
4. **Fusion (P148, wiedereröffnet):** Die Aussage „unter dem Gate-Rauschen“ war
   eine OBergrenzen-Schätzung aus dem ALTEN Budget-15-Trace (467 Dispatches),
   keine Messung am integrierten Budget-3-Stand. Ein eigener Dispatch-Trace des
   Draft-Pfads steht aus; bis dahin ist P148 eine offene Frage, kein befundenes
   „nutzt nichts“.

## Konsequenz für das Ziel „mindestens so schnell wie der alte Drafter"

Der Schritt besteht aus Trunk-Steuer (≈16 ms, identisch für jeden Drafter) plus
Drafter-Eigenanteil (draft 6.8 + gdn 1.6 ≈ 8.4 ms). Der alte 51248ae-Drafter
zahlt weniger Drafter-Eigenanteil (rund 1.4 ms laut seinem Commit) und weniger
Verify-Zeilen pro Update (höhere Annahme, 3.4 vs 1.5–2.6 tok/update). Die
verbleibende Differenz (code 92 vs 168 tok/s) zerfällt in:

- **Annahmelänge** (2.6 vs 3.4 tok/update): Checkpoint-Qualität, Training.
  Nicht durch Kernel-Arbeit behebbar. [INFERENCE] — die Kernel können die
  Annahme nicht erhöhen, nur deren Kosten senken; senken ist geschehen.
- **DLoop-Eigenanteil** 8.4 ms/Schritt: die `gdn_*`-Kernel (1.6 ms) sind der
  letzte DLoop-Kern-Hebel; draft-Intern (6.8 ms) ist nach Welle 1
  dispatch-/latenzgetrieben.

**Nächster Schritt (P149):** `gdn_recurrent_tree`, `gdn_tree_commit`,
`gdn_conv_tree` im eigenen Worktree optimieren (94 284 Aufrufe, 4.1 s gesamt,
Ziel ≤ 0.8 ms/Schritt). Die Trunk-Seite bleibt offen: `hc_read` ist Rechenlast
mit Fusion-/Auslastungsfragen (P150, erst nach Messung des gewählten Kernels),
kein Naturgesetz. Der Annahme-Anteil der Restdifferenz ist Training.