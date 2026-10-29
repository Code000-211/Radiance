# Neue Quantisierungsmethoden 2026: arXiv und GitHub (Stand 2026-10-29)

Quellen: arXiv-API und Volltexte (arxiv.org/html) waren erreichbar; github.com-Seiten gesperrt, Raw-Dateien
von raw.githubusercontent.com erreichbar. Zahlen aus Abstracts oder Volltexten, Herkunft jeweils angegeben.

## Trellis und Codebooks

- **XOR-Trellis** (arXiv 2610.00432, 2026-09-30): rotationsfreie Trellis-Quantisierung. Ein strukturierter
  Zustand-zu-Wert-Mapper macht die Rekonstruktion billig, eine krümmungsbewusste Pfadsuche ersetzt die
  Hadamard-Rotation. Llama-3.1-8B, WikiText-2: PPL 8.00, gegen QTIP-3INST mit Hadamard 8.48 und ungedrehter
  Trellis 8.69. Gemessen bei etwa 2.25 bit (FP4-Decoder, E5M3-Skala je 32 Gewichte). Kein Code gefunden.
  Relevanz: Ohne Hadamard entfällt der Zwang zu 128er-Blöcken für TP. Belegt nur bei 2.25 bit.
- **BCJR-QAT** (arXiv 2605.10655, 2026-05-11, Code: github.com/Venugopalan2610/quant-2bit): ersetzt das
  Viterbi-Argmax durch BCJR-Vorwärts-Rückwärts, dadurch differenzierbar. Llama-3.2-1B, Schicht 4, 2 bit,
  End-to-End-Distillation mit Forward-KL: −0.084 PPL gegen QTIP-PTQ. Fused Triton-Kernel, 6.57× schneller.
  Relevanz: widerspricht der früheren Annahme, Trellis-Codes ließen sich nicht per Gradient anpassen.
  Forschungscode, 1B-Modell, 2 bit, PPL statt KL.
- **ik_llama.cpp KT-Quants** (github.com/ikawrakow/ik_llama.cpp README, Abschnitt Trellis): IQ1_KT bis IQ4_KT,
  Trellis mit ganzzahliger Basis, CPU, CUDA, Metal und Neon. Keine RDNA-Angaben, keine KL-Zahlen gelesen.
  Korrektur: Im Mainline-llama.cpp gibt es kein Trellis-Format, in ik_llama.cpp aber schon.
- **BARQ** (arXiv 2610.04490, 2026-10-03, Code: github.com/chenhangcuisg-code/BARQ): ausgewogene Codebuch-
  Verfeinerung per optimalem Transport. Nur 2 bis 2.5 bit pro Wert, gegen GPT-VQ −7 % PPL.
- **MPQ** (arXiv 2610.32127, 2026-10-23): Message-Passing statt hartem GPTQ/LDLQ. Bei 4 bit sind die Basislinien
  nahe BF16, der Gewinn ist klein; größer bei 2 bis 3 bit. Die Autoren halten eine Erweiterung auf QTIP-Codes
  für möglich, haben sie nicht getestet.

## Skalare und Mikroskalen-Formate bei 4 bit (kein Trellis)

- **AAAC** (arXiv 2605.08692, 2026-05-09): zwei gelernte skalare Codebücher je Schicht (64 Byte), Auswahl je
  Gruppe über das Vorzeichenbit der Skala, ohne Speicheraufschlag. Vergleich mit AWQ/GPTQ bei 4 bit.
- **Grid Games** (arXiv 2605.12327): je Gruppe zwischen zwei 4-bit-Gittern wählen, Gewinn bei NVFP4/MXFP4.
- **SOAR** (arXiv 2605.12245, 2026-05-12): gemeinsame Optimierung globaler und blockweiser NVFP4-Skalen.
- **CubicQuant** (arXiv 2608.06763): parametrische nichtuniforme Skalar-Gitter, 1 bis 8 bit.
- **any4** (arXiv 2507.04610, 2025-07): gelerntes 4-bit-Zahlenformat, kalibriert mit einem kuratierten Sample.

Einordnung: Alle fünf sind skalar und liegen nach den Gauß-Schranken-Zahlen des Berichts 1.2 bis 1.5 dB
hinter einem Trellis. Sie sind Alternativen für Hardware mit festen Formaten, nicht für EXL3.

## Rundung und Rotation

- **ADMM-Q** (arXiv 2605.11222, 2026-05-11): Hessian-basierter Gewichtsquantisierer mit ADMM, Ziel unter 4 bit,
  Alternative zu GPTQ und LDLQ. Keine Zahlen gelesen.
- **OptRot** (arXiv 2512.24124, 2025-12-30): datenfreie Rotation durch Minimierung der vierten Potenz der
  rotierten Gewichte; schlägt Hadamard und SpinQuant für GPTQ. Für EXL3 ungeeignet, weil die Transformation
  fest im Format steht.
- **PolarQuant** (arXiv 2603.29078, 2026-03-30): Hadamard plus Gauß-Zentroide, ohne Kalibrierung; Qwen3.5-9B,
  PPL 6.40 gegen 6.90 bei absmax Q5.

## Hybride mit Gated DeltaNet

- **Why Gated DeltaNet Survives 4-Bit Quantization** (arXiv 2609.04098, 2026-09-03): NVFP4 W4A4 auf Qwen3.8-27B,
  bereits im Bericht als Minima-Messung berücksichtigt.
- Rekurrenter Zustand, nicht Gewichte: LeapQuant (2609.38166), DAMP (2608.27513), STEPQuant (2609.38169),
  Preserving Updates (2610.15709). Betreffen die Quantisierung des Zustands zur Inferenzzeit, nicht die Bits
  der Gewichte.

## Kalibrierung

- **Beyond 4 bits** (arXiv 2610.28537, 2026-10-20): bei 2 bit ist die Darstellung der größte Faktor; die
  Erholungsdaten müssen zu den Kalibrierdaten passen. Stützt die In-Domain-Kalibrierung aus dem Bericht.

## Was daraus für den Plan folgt

- Keine der neuen Arbeiten zeigt bei 4 bit einen Gewinn gegenüber EXL3 mit YAQA. Die Berichtsempfehlung bleibt.
- Am ehesten relevant sind XOR-Trellis (Rotationsfreiheit für TP) und BCJR-QAT (Gradienten durch Trellis-Codes).
  Beide wären eigene Projekte; keines liegt als lauffähiger Code für EXL3 vor.
- Die Aussage „Trellis-Codes sind nicht gradientenfähig" aus den Notizen zum Feintuning ist durch BCJR-QAT überholt.
