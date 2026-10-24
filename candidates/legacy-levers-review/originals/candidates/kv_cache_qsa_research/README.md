# KV-Cache und QSA: Recherche zu Kandidaten (Stand 2026-10-08)

**Art:** Recherchebericht, kein Hebel-Paket. Es gibt keinen `lever.patch` und kein `gate.json`;
dieses Verzeichnis gehört nicht in die Lever-Queue. Es sammelt Kandidaten mit Quellen und schlägt
die Diagnosen vor, die vor jeder Kernel-Arbeit stehen sollten.

**Regeln für das Lesen dieses Berichts (CLAUDE.md §1):**

- Zahlen aus Papers, Blogs und READMEs sind **Angaben der jeweiligen Quelle**, nicht von uns gemessen.
  Bei jeder steht, wer sie gemessen hat und ob es eine unabhängige Bestätigung gibt.
- Zahlen zu unserer Engine sind entweder aus dem Code belegt (mit Pfad) oder als **Rechnung**
  gekennzeichnet. Gemessen haben wir für diesen Bericht nichts.
- Viele Primärquellen (arxiv.org, vllm.ai, rocm.blogs.amd.com, research.samsung.com) waren aus der
  Recherche-Umgebung nicht erreichbar. Diese Angaben stammen aus Suchzusammenfassungen, Abstracts
  oder GitHub-READMEs. Im Quellenverzeichnis steht je Quelle, was tatsächlich gelesen wurde.
  Vor einer Umsetzung sind die Tabellen der Primärquelle zu prüfen.

---

## 1. Ausgangslage im Code

| Fakt | Beleg |
|---|---|
| KV-Cache ist FP8 E4M3 oder BF16; andere Formate gibt es nicht | `core/config.cpp:60`, `:398` (`--kv-cache-dtype` nimmt nur `bf16 \| fp8`) |
| FP8-Store mit festem Descale je Aufruf, 1.0 ist der Standard | `libr4d/r4d.h` Kommentar zu `r4d_kv_store_fp8` |
| Attention-Kernel kompiliert für `head_dim` 256, GQA 6, Paged-Block 16 | `libr4d/r4d.h` (Kopfkommentar, `r4d_attn_decode_h256_gqa6_fp8kv`) |
| 12 Full-Attention-Schichten (jede vierte), dazwischen GDN | `docs/QSA.md` („every fourth: 3, 7, … 47“, „12 QSA layers“) |
| 2 KV-Heads × 256; unter TP2 hält jeder Rang einen KV-Head | `docs/QSA.md`, `arch/common/rad_block_attn_gated_fp8.h:186–191` |
| RoPE nur auf 64 von 256 Dimensionen (`partial_rotary_factor` 0.25), `k_norm` vor dem Cache | `docs/QSA.md`, `arch/common/rad_block_attn_gated_fp8.h:228–230` |
| QSA: Decode liest die besten 512 Blöcke à 4 Tokens plus Rest, also höchstens 2051 Tokens | `docs/QSA.md` („The exactness bound“) |
| **Der Indexer (`qsa_score`, `qsa_select`) ist der einzige Teil eines Decode-Schritts, der mit dem Kontext wächst** | `docs/QSA.md` („What the indexer costs“) |
| Zwei Implementierungen von `qsa_select` müssen dieselben **Bytes** liefern (Reihenfolge der Seitentabelle), nicht nur dieselbe Menge | `docs/QSA.md` |
| Prefix-Caching für Hybridmodelle mit GDN-Checkpoints existiert bereits | `core/mem/prefix.h` |
| KV-Pool 1,56 GiB (`--vram-kv-mib 1600`), Pinned-Host-Pool 12 GiB | `HARDWARE_GROUND_TRUTH.md` (Speicheraufteilung) |
| VRAM-Kopierkernel ~545 GB/s; Peer-Reads über PCIe ~4,7× langsamer als Peer-Writes | `HARDWARE_GROUND_TRUTH.md` |
| MTP-Kopf hat einen eigenen QSA-Indexer | `arch/common/rad_block_mtp_hc.h:151–183`, `:300` |

### Korrektur zu früheren Annahmen

**KVarN ist auf `main` nicht gebaut.** `arch/common/rad_block_attn_paged_fp8.h:107` lehnt den
Pfad ab („kv_csa3_kvarn_pool kernel ungrounded in binary, see postmortem“).
`libr4d/r4d_kv_anchor_pool4.hip` wird laut
`docs/history/HOW_HALLUCINATIONS_CORRUPTED_THE_WORKTREE_AND_CONTEXT_MODE_POSTMORTEM.md` nicht
gebaut. `candidates/lever3_qsa_offload_kvarn/` beschreibt ein INT4-KV und einen Offload, die nie als
Kernel existiert haben. Die dort genannte „4× Kompression“ ist gegen BF16 gerechnet, nicht gemessen.
Der Name KVarN gehört inzwischen zu einem echten Paper (Huawei, Juni 2026, Abschnitt 3.2); das ist
nicht dasselbe wie der frühere Kandidat.

### Rechnungen (nicht gemessen)

- KV je Token und GPU heute: 12 Schichten × 1 KV-Head × 256 × 2 (K, V) × 1 Byte = **6 KiB**.
- Kapazität des 1,56-GiB-Pools, falls er je GPU gilt: ~**270k Tokens** über alle Sequenzen.
- Decode-Lesen der Attention je Sequenz und Schritt: 2051 × 6 KiB ≈ **12 MiB** je GPU,
  bei ~545 GB/s ≈ **23 µs**. Mit 4-Bit-KV ≈ 7 MiB ≈ 13 µs.
- Block-Keys des Indexers: 64 B je Token und QSA-Schicht (`docs/QSA.md`). Bei 32k Kontext sind
  das 12 × 2 MiB = 24 MiB je Sequenz und Schritt, also **mehr als die Attention selbst liest**.
  Ob die Block-Keys je Rang repliziert sind, ist für diese Rechnung nicht geprüft.

Daraus folgt die Grundthese dieses Berichts: Bei langem Kontext liegt der wachsende Anteil im
Indexer, nicht im KV-Lesen der Attention. KV-Quantisierung ist bei uns eher ein Kapazitätshebel
als ein Geschwindigkeitshebel. Beides ist durch Messung zu bestätigen (Abschnitt 5).

---

## 2. Teil A: KV-Quantisierung

### 2.1 Kandidaten

Belegklassen: **U** = unabhängig gemessen (nicht von den Autoren), **H** = Hersteller oder
Framework-Team misst die eigene Implementierung, **A** = nur die Autoren, **P** = Preprint ohne
Zahlen, die ich gesehen habe.

| Verfahren | Bits je Wert | Gegen unser FP8 | Belege | Bewertung |
|---|---|---|---|---|
| FP8 E4M3 (heute) | 8 | 1× | U: vLLM-Studie: >98 % Accuracy-Recovery, Empfehlung als Standard. SGLang, Qwen3-235B: AIME25 0,773 (BF16) → 0,733 (FP8) | Status quo |
| TurboQuant k8v4 (K in FP8, V 4 Bit) | ~6 | ~1,3× | U: vLLM-Studie, >98 % Recovery wie FP8 | Sicherste Stufe, wenig Gewinn |
| **TurboQuant 4bit-nc** (Hadamard, Lloyd-Max-Codebuch, exakte Norm) | ~4,5 | ~1,8× | U: vLLM-Studie, 96 % Recovery, ~20 % mehr Latenz. U auf gfx1201 (RX 9070 XT, llama.cpp-Fork): Decode 49,3 statt 53,4 t/s (f16), PPL +0,6 %, auch an Qwen3.5-9B (`head_dim` 256, hybrid) getestet | **Bester 4-Bit-Kandidat** |
| TurboQuant 3bit-nc, k3v4-nc | ~3,5 | ~2,3× | U: vLLM-Studie, etwa −20 Punkte bei Reasoning, als nicht produktionstauglich eingestuft | Nein |
| KVarN (Hadamard plus Varianz-Normierung auf beiden Achsen, K4V2, Gruppe 128) | ~2,3–3 | ~3× | A: GLM-4.7-Flash AIME25 53,3 % gleich BF16, Kapazität 2,77×, Durchsatz 0,94×. Hybrid unterstützt (Qwen3.6-27B). Keine Replikation gefunden | Interessant, unbelegt. Kachel = vLLM-Block (128/64 Tokens) mit fp16-Puffer; unser Block ist 16 |
| FP4 (E2M1 / NVFP4) | 4 | 2× | H: NVIDIA <1 % Verlust (NVFP4, Blackwell). U: SGLang, Qwen3-235B AIME25 → 0,600 mit FP4 | Nein. Nach meinem Kenntnisstand ohne FP4-Hardware auf RDNA4 (nicht im Repo geprüft) |
| LittleBit (Faktorisierung, binäre Faktoren, Sub-1-Bit) | 0,1–1 | — | A: für **Gewichte**, braucht QAT | Nicht auf KV übertragbar; bei 4 Bit nicht sinnvoll |

### 2.2 Fehlerkorrektur: was davon trägt

| Verfahren | Idee | Belege | Bewertung |
|---|---|---|---|
| Exakte Norm je Vektor („nc“) | Norm speichern, Rekonstruktion darauf skalieren | U: im ausgelieferten vLLM-Preset | Billig, übernehmen |
| Keys mehr Bits als Values | K ist empfindlicher (Ausreißer-Kanäle, Softmax) | U: vLLM k8v4; A: KVarN K4V2; A: „More for Keys, Less for Values“ | Übernehmen |
| Neueste Tokens unkomprimiert | Puffer, bis eine Gruppe voll ist | Bei GEAR, KVarN, KIVI so gebaut | Bei uns z. B. der QSA-Rest und der laufende 16er-Block |
| QJL-Restfehler (TurboQuant Stufe 2) | 1-Bit-Skizze des Restfehlers, unverzerrtes Skalarprodukt | U: Unverzerrtheit synthetisch bestätigt, aber ~42 % relativer Fehler je Score bei d = 128. Die ausgelieferten Implementierungen, die ich gesehen habe, nennen QJL nicht | In der Praxis weggelassen |
| GEAR | Low-Rank- plus Sparse-Restfehler | A: Zahlen unterscheiden sich zwischen Paper-Versionen; SVD je Puffer | Kein ausgelieferter Kernel bekannt |
| HeadQ | Low-Rank-Restfehler der Keys in Query-PCA-Basis, additive Logit-Korrektur | P: Mai 2026, keine Benchmark-Zahlen gesehen | Passt gut zur Struktur. Wichtigster Gedanke: Fehler an Logits/Attention-KL messen, nicht an der Rekonstruktion |
| KVLinC | Hadamard für V, trainierte lineare Korrektur für K, 2 Bit | A: 2,55× gegen FlashAttention, eigene Messung | Braucht Training |
| CommVQ | Additive VQ, Codebuch kommutiert mit RoPE | A: ICML 2025, 2 Bit und 1 Bit | Setzt volles RoPE voraus, wir haben partielles |
| AQUA-KV | K/V aus der Vorgängerschicht vorhersagen, Rest quantisieren | A: ICML 2025, 2–2,5 Bit | Bei uns liegen 3 GDN-Schichten zwischen zwei Attention-Schichten, die Annahme ist fraglich |
| QuantSpec | INT8-KV als oberes Nibble plus quantisierter Restfehler; Draft liest nur das obere | A: ICML 2025, >90 % Akzeptanz, bis 2,5× | Nutzt das ganze Modell als Draft; unser Draft ist der MTP-Kopf. Wenig Passung |

### 2.3 Geschwindigkeitsangaben zu TurboQuant, eingeordnet

| Angabe | Quelle | Basis des Vergleichs |
|---|---|---|
| „bis zu 8×“ | Google, laut Presseberichten | nur Logit-Berechnung, 4 Bit gegen **32-Bit**-Keys, H100 |
| 8,4× Decode | Community-Plugin (varjoranta) | Basis mit `enforce_eager`; die Autoren selbst nennen es nicht vergleichbar |
| bis ~3,6× | AMD ROCm-Blog | gegen die Open-Source-TurboQuant-Implementierung von vLLM, nicht gegen FP8/BF16 |
| −40 bis −52 % Durchsatz bei 2,3–3,7× Kapazität | vLLM-Studie, Qwen3-30B-A3B | gegen FP8 |
| −8 % Decode | llama.cpp-Fork, gfx1201 | gegen f16 |

Schneller wird es nur, wenn das Lesen des KV-Caches den Decode dominiert oder wenn der Batch durch
KV-Kapazität begrenzt ist. Das Erste trifft bei uns wegen FP8 und QSA kaum zu (Rechnung oben). Das
Zweite ist offen.

---

## 3. Teil B: Rund um QSA und die KV-Verwaltung

### 3.1 Top-k-Auswahl aus dem vorigen Schritt als Startwert (exakt)

- **Quelle:** TensorRT-LLM „Guess-Verify-Refine“ (GVR) für DSA. Die Indizes des vorigen Schritts
  dienen als Startwert für die Top-k-Schwelle, danach wird geprüft und nachgebessert. Das Ergebnis
  ist exakt. ScoutAttention (A) misst, dass sich im Schnitt weniger als 15 % der wichtigen Blöcke
  zwischen zwei Tokens ändern. Bei HiSparse (A) verfehlt die reine Top-k-Menge des vorigen Schritts
  ~30 % der neuen Auswahl.
- **Für uns:** beschleunigt `qsa_select` ohne Paritätsrisiko, **wenn** die Seitentabelle
  byte-identisch bleibt (`docs/QSA.md`).
- **Grenzen:** ausgeliefert nur für Blackwell und bestimmte k. Spart nur die Auswahl; `qsa_score`
  liest weiterhin alle Block-Keys.
- **Engine-Regeln:** eigene Op oder eigener Kernel für eine bestehende Op im eigenen Plugin,
  über die Hierarchie ausgewählt; `libref`-Orakel für `qsa_select` existiert.

### 3.2 Indexer-Auswahl über Schichten teilen (verlustbehaftet, großer Hebel)

- **Quellen:**
  - IndexCache (Tsinghua/Z.ai): Nur ~1/4 der Schichten rechnen den Indexer, die anderen übernehmen
    die Indizes. Laut eigenen Zahlen, 30B-DSA-Modell auf H100 bei 200k Kontext: Decode 58 → 86 tok/s
    (1,48×), Prefill 1,82×. Überlappung benachbarter Schichten 70–100 %. Qualität nur qualitativ
    („virtually unchanged“). Trainingsfreie Variante per gieriger Suche auf Kalibrierdaten.
  - Kascade (Microsoft): trainingsfrei, Anker-Schichten per DP gewählt.
  - GLM-5.2 IndexShare: dasselbe in Produktion, aber **mittrainiert**.
- **Für uns entscheidend:** DSA hat in **jeder** Schicht einen Indexer. Unsere 12 QSA-Schichten
  liegen 4 Schichten auseinander, mit GDN dazwischen. Ob die Überlappung dann noch hoch ist, ist
  völlig offen.
- **Engine-Regeln:** verlustbehaftet, verändert die Ausgabe gegenüber den Goldens. Ob das in die
  Gate-Toleranzen passt, entscheidet Leon.

### 3.3 Indexer über einen gemeinsamen Prefix im Batch kacheln (exakt, abhängig von der Last)

- **Quellen:** Hydragen (ICML 2024), FlashInfer Cascade. Die Attention auf einen
  gemeinsamen Prefix läuft über alle Sequenzen gebündelt (GEMM statt GEMV). Die großen Zahlen
  (>30×) gelten bei 16k–32k langen Prefixes, Batch 64–256 und schwacher Basis. Die
  Hydragen-Autoren schreiben selbst: Wenn das Lesen der Gewichte dominiert, bringt es wenig.
- **Für uns:** Die Decode-Attention liest je Sequenz andere Blöcke, dort bringt es wenig. Aber
  `qsa_score` über die Block-Keys eines gemeinsamen Prefixes ließe sich über Sequenzen kacheln. Jeder
  Score ist ein eigenes Skalarprodukt; mit unveränderter Summationsreihenfolge kann das
  bit-identisch bleiben. `qsa_score` kachelt heute schon je Sequenz (`docs/QSA.md`).
- **Voraussetzung:** BetterBench muss viele Sequenzen mit langem gemeinsamem Prompt fahren.
  Nicht geprüft.

### 3.4 KV-Cache in den Host auslagern, nur die Auswahl liegt im VRAM (Kapazität)

- **Quellen:** HiSparse (Stanford, arXiv 2608.07009), ausgeliefert in SGLang für DeepSeek-V4 und
  GLM-5.1, eine vLLM-Variante gibt es für GLM-5.3. Die ganze KV-Historie liegt im Host, auf der GPU
  ein LRU-Cache von ~2× top-k. Laut Paper (A) 87 % Trefferquote auf einer GLM-5.1-Spur. Die Ausgaben
  bleiben gleich, weil sich nur der Speicherort ändert. Laut LMSYS (H) >3× Durchsatz bei 256
  parallelen Anfragen. **Bei wenigen parallelen Anfragen verlangsamt es** (LMSYS). In SGLang nur mit
  Prefill/Decode-Disaggregation.
- **Unsere Hardware spricht dagegen:**
  - **Lesen über PCIe ist langsam:** Liest der Kernel fehlende Blöcke selbst aus dem Host, ist das
    der langsame Lesepfad (`HARDWARE_GROUND_TRUTH.md`, ~4,7× langsamer als Schreiben).
  - **DMA-Kopien sind im Plugin verboten:** Plugins dürfen außer `hipLaunchKernel` keine
    HIP-Stream-APIs nutzen (`docs/IMPLEMENTATION.md`).
  - **Fehltreffer kosten spürbar Zeit (Rechnung, nicht gemessen):** 13 % von 512 Blöcken × 4 Tokens
    × 512 B × 12 Schichten ≈ 1,6 MiB je Sequenz und Schritt, beim Lesen über PCIe einige 100 µs.
- **Vorgeschichte:** `lever3_qsa_offload_kvarn` war schon einmal ein solcher Offload. Laut
  Postmortem war er halluziniert, unter anderem mit „nur 2 von 48 Schichten“.

### 3.5 Indexer-Auswahl über MTP-Draft-Schritte wiederverwenden

- **Quelle:** TensorRT-LLM bietet eine Option, die Indexer-Top-k über die MTP-Draft-Schritte
  wiederzuverwenden (H, Konfigurationsdoku).
- **Für uns:** Der MTP-Kopf hat einen eigenen Indexer (`arch/common/rad_block_mtp_hc.h`). Der
  Gewinn ist vermutlich klein, nur ein Kopf. Verlustbehaftet für den Draft; die Verifikation bleibt
  exakt.

---

## 4. Was für jeden Kandidaten gilt (Engine-Regeln)

- Keine Änderung an Op-Schemas. Ein neues KV-Format oder eine neue Auswahl-Variante ist eine
  **neue Op** mit `libref`-Zeile, damit `rad-kbench` und das Orakel sie prüfen können (`spec.md` §17,
  `docs/OPS.md`).
- Modellspezifische Kernel gehören in unser eigenes Plugin, nicht in `libr4d` (`docs/UPSTREAM.md`).
- Ein neuer Wert für `--kv-cache-dtype` oder ein Flag braucht `candidates/<slug>/gate.json`.
- Kein Python in Engine, Build oder CTest. Analyse-Werkzeuge gehören nach `scripts/`.
- Verlustbehaftete Verfahren (2.1, 3.2, 3.5) ändern die Parität. Die Toleranzen in `baselines/`
  ändert nur Leon.
- Diagnose-Ops, die K/V zur Messung rund-quantisieren, laufen nie im Gate und nie mit
  `--debug-accept-reference-kernels`.

---

## 5. Vorgeschlagene Reihenfolge

### Diagnosen zuerst (keine Kernel-Arbeit, keine Paritätsänderung)

- **D1, wo die Decode-Zeit bei BetterBench-Kontextlänge hingeht:** Anteil von `qsa_score` plus
  `qsa_select` und Anteil der Attention am Schritt. Werkzeuge: `scripts/profile_engine.py`,
  `r4d_selftest --perfqsa` (`docs/QSA.md`), Trace nach `libr4d/isa/trace.sh`.
- **D2, ob die KV-Kapazität den Score begrenzt:** `kv_util` und
  `radiance:prefix_cache_evictions_total` unter BetterBench (`core/mem/prefix.h`, „WHAT TO WATCH“).
- **D3, wie stabil die QSA-Auswahl ist:** Auswahl je QSA-Schicht und Schritt mitschreiben, dann
  die Überlappung messen: zwischen QSA-Schichten (für 3.2), zwischen Schritten (für 3.1) und
  zwischen Sequenzen mit gemeinsamem Prefix (für 3.3). Ausgabe über den Diagnose-Modus
  (`docs/DIAGNOSE.md`).
- **D4, nur falls D2 zeigt, dass Kapazität begrenzt:** Accuracy der KV-Formate messen, mit einer
  Diagnose-Variante von `kv_store`, die K/V durch das Zielformat rundet und als FP8/BF16 schreibt.
  Die Attention bleibt unverändert. Verglichen werden FP8, k8v4, K4V4-nc und K4V2 (KVarN-Stil),
  gemessen an der Attention-KL und an den Benchmarks, nicht an der MSE der Rekonstruktion (HeadQ).

### Danach, je nach Ergebnis

1. Ist der Indexer teuer: zuerst die exakten Hebel 3.1 und 3.3.
2. Ist die Überlappung über Schichten hoch (D3): 3.2 als verlustbehafteter Kandidat; die Parität
   entscheidet Leon.
3. Begrenzt die Kapazität (D2) und ist die Accuracy akzeptabel (D4): 4-Bit-KV nach TurboQuant
   4bit-nc oder k8v4. Der Decode-Kernel entpackt im Kernel, dreht q einmal und V am Ende zurück;
   `libr4d/r4d_fwht.h` ist vorhanden.
4. 3.4 (Host-Offload) erst, wenn die beiden Hardware-Hindernisse aus 3.4 gelöst sind.

---

## 6. Offene Fragen an Leon

1. Wie groß sind Kontext, Parallelität und gemeinsamer Prefix bei BetterBench? Davon hängen 3.3,
   3.4 und Teil A ab.
2. Sind verlustbehaftete Hebel (3.2, 4-Bit-KV) grundsätzlich zulässig, und mit welcher Toleranz?
3. Gilt der 1,56-GiB-KV-Pool je GPU (Annahme in den Rechnungen)?

---

## 7. Quellen

Spalte „gelesen“: **R** = README/Seite direkt gelesen, **S** = nur Suchzusammenfassung oder
Abstract, **B** = Seite war in der Recherche-Umgebung gesperrt (Inhalt nur über S).

### KV-Quantisierung

| Quelle | gelesen | Inhalt |
|---|---|---|
| [TurboQuant, arXiv 2504.19874 (ICLR 2026)](https://arxiv.org/html/2504.19874v1) | B/S | Methode, QJL-Restfehler |
| [vLLM: A First Comprehensive Study of TurboQuant (2026-05-11)](https://vllm.ai/blog/2026-05-11-turboquant) | B/S | Accuracy und Durchsatz gegen FP8 |
| [vLLM: The State of FP8 KV-Cache](https://vllm.ai/blog/2026-04-22-fp8-kvcache) | S | FP8 als Standard |
| [vLLM PR #38479 (TurboQuant-Backend)](https://github.com/vllm-project/vllm/pull/38479) | S | Implementierung: WHT plus Lloyd-Max für K, uniform für V |
| [vLLM-Forum zu TurboQuant](https://discuss.vllm.ai/t/vllm-turboquant-implementation/2944) | S | FP8 bleibt Empfehlung |
| [llama.cpp-turboquant für ROCm (gfx1201)](https://github.com/Pascal-SAPUI5/llama.cpp-turboquant) | R | Decode- und PPL-Zahlen auf gfx1201 |
| [llama.cpp Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) | S | Community-Tests |
| [llama.cpp Issue #21385](https://github.com/ggml-org/llama.cpp/issues/21385) | S | q4_0-KV auf Qwen3.5-9B |
| [zededa: TurboQuant auf Jetson Thor](https://zededa.com/?p=10018) | S | unabhängiger QJL-Test |
| [InfoQ zu TurboQuant](https://infoq.com/news/2026/04/turboquant-compression-kv-cache) | S | Einordnung |
| [Tom's Hardware zu TurboQuant](https://tomshardware.com/tech-industry/artificial-intelligence/googles-turboquant-compresses-llm-kv-caches-to-3-bits-with-no-accuracy-loss) | S | „8×“-Angabe |
| [varjoranta/turboquant-vllm](https://github.com/varjoranta/turboquant-vllm) | S | 8,4×-Angabe und ihre Einschränkung |
| [AMD: Productionizing TurboQuant on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/turboquant-vllm-agentic/README.html) | B/S | ~3,6× gegen die vLLM-TurboQuant-Implementierung |
| [sotaaz: TurboQuant-Stand August 2026](https://sotaaz.com/post/turboquant-status-2026-en) | S | Stand der Implementierungen |
| [KVarN, arXiv 2606.03458](https://arxiv.org/pdf/2606.03458) | B/S | Methode |
| [huawei-csl/KVarN](https://github.com/huawei-csl/KVarN) | R | Presets, Modelle, GLM-Zahlen |
| [NYU Shanghai zu KVarN](https://rits.shanghai.nyu.edu/ai/huaweis-kvarn-hits-5x-kv-cache-capacity-at-fp16-accuracy-and-throughput) | S | Sekundärbericht |
| [SharpInference: KVarN-Port](https://github.com/pekkah/SharpInference/issues/180) | S | Effektive Bitrate ~2,3 |
| [GEAR (PMLR, NeurIPS 2024)](https://proceedings.mlr.press/v262/kang24a.html) | S | Low-Rank- plus Sparse-Restfehler |
| [HeadQ, arXiv 2605.03562](https://arxiv.org/html/2605.03562v1) | B/S | Logit-Korrektur in Query-Basis |
| [KVLinC, arXiv 2510.05373](https://arxiv.org/html/2510.05373v1) | B/S | Hadamard plus lineare Korrektur |
| [AQUA-KV, arXiv 2501.19392](https://arxiv.org/html/2501.19392v4) | B/S | Prädiktor über Schichten |
| [CommVQ, arXiv 2506.18879](https://arxiv.org/html/2506.18879v1) | B/S | RoPE-kommutatives Codebuch |
| [Kitty, arXiv 2511.18643](https://arxiv.org/pdf/2511.18643) | B/S | 2-Bit-KIVI verliert bei Reasoning |
| [More for Keys, Less for Values, arXiv 2502.15075](https://arxiv.org/html/2502.15075v3) | B/S | K braucht mehr Bits als V |
| [QuantSpec, arXiv 2502.10424](https://arxiv.org/html/2502.10424v1) | B/S | Hierarchischer INT8 = INT4 plus INT4-Restfehler |
| [SGLang: Quantized KV Cache](https://docs.sglang.io/advanced_features/quantized_kv_cache.html) | S | AIME25 BF16/FP8/FP4 |
| [NVIDIA: NVFP4 KV-Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/) | S | Herstellerangaben |
| [Qwen3.5 Hybrid-Analyse (ai-muninn)](https://ai-muninn.com/en/blog/turboquant-kv-cache-benchmark-gx10) | S | Nur 1/4 der Schichten hat KV |
| [LittleBit, arXiv 2506.13771](https://arxiv.org/html/2506.13771v5) | B/S | Sub-1-Bit-Gewichte |
| [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit) | R | README |

### QSA, Indexer, KV-Verwaltung

| Quelle | gelesen | Inhalt |
|---|---|---|
| [THUDM/IndexCache](https://github.com/THUDM/IndexCache) | R | Varianten, Speedups, Überlappung |
| [IndexCache, arXiv 2603.12201](https://arxiv.org/html/2603.12201v1) | B/S | Paper |
| [Kascade, arXiv 2512.16391](https://www.arxiv.org/pdf/2512.16391) | B/S | Trainingsfreies Teilen über Schichten |
| [GLM IndexShare (Raschka)](https://sebastianraschka.com/llm-architecture-gallery/indexshare/) | S | Produktion, mittrainiert |
| [Guess-Verify-Refine, arXiv 2604.22312](https://arxiv.org/pdf/2604.22312) | B/S | Top-k mit Startwert aus dem vorigen Schritt |
| [TensorRT-LLM DeepSeekV4SparseAttentionConfig](https://nvidia.github.io/TensorRT-LLM/llm-api/reference/DeepSeekV4SparseAttentionConfig.html) | S | GVR, Wiederverwendung über MTP-Schritte |
| [ScoutAttention, arXiv 2603.27138](https://arxiv.org/html/2603.27138) | B/S | <15 % Blockwechsel je Token |
| [PIVOT, arXiv 2607.24593](https://arxiv.org/pdf/2607.24593) | B/S | Indexer-Scan über benachbarte Queries teilen |
| [HiSparse, arXiv 2608.07009](https://arxiv.org/html/2608.07009v1) | B/S | Host-Offload mit GPU-LRU |
| [SGLang HiSparse-Guide](https://docs.sglang.io/advanced_features/hisparse_guide.html) | S | Einschränkungen (PD-Disaggregation) |
| [LMSYS: SGLang HiSparse](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse) | S | >3× bei 256 Anfragen, Overhead bei wenigen |
| [LMSYS: DeepSeek-V4 Day 0](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) | S | HiSparse für V4 |
| [vLLM: GLM 5.3 Hybrid HiSparse Offloading](https://vllm.ai/blog/2026-09-07-glm53-part1-hybrid-sparse-offloading) | B/S | vLLM-Variante |
| [SparDA, arXiv 2606.04511](https://arxiv.org/pdf/2606.04511) | B/S | Vorhersagendes Prefetching |
| [FlexiCache, arXiv 2511.00868](https://arxiv.org/html/2511.00868v1) | B/S | Stabile Heads, Top-K-Seiten im VRAM |
| [Hydragen, arXiv 2402.05099](https://arxiv.org/html/2402.05099v2) | B/S | Gemeinsamer Prefix im Batch |
| [FlashInfer, arXiv 2501.01005](https://arxiv.org/pdf/2501.01005) | B/S | Cascade-Attention |
