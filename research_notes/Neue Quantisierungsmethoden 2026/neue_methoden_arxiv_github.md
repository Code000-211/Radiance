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

## Nachtrag 2026-11-01: Gibt es ein besseres Format als einen Trellis bei 4 bis 4.5 bpw?

Quellen: arXiv-API und Volltexte (arxiv.org/html).

- **HyperQuant** (arXiv 2606.23406, 2026-06-22): Hadamard, Gitter E8/D4/A2/Z, verlustfreies Bit-Stripping und
  Rice-Codierung variabler Länge. Llama-3.1-8B-Instruct, WikiText-2, bei 4.0 bit pro Skalar: HIGGS 7.618 PPL,
  HyperQuant (2D-Gitter) 7.500. Kein Vergleich mit QTIP. Variable Länge braucht eigenen GPU-Decoder.
- **LLVQ, Leech-Gitter** (arXiv 2603.11021, Qualcomm): schlägt QuIP#, QTIP und PVQ, Ergebnisse bei 2 bit
  (Llama-2-7B). Nachfolger **Tetra** (2609.35465) bei 2.7 bpw über das ganze Modell.
- **CoS-VQ** (arXiv 2610.37336, 2026-10-27): gemeinsame Optimierung von Indizes und geteilter Skala; schlägt QTIP
  und LLVQ, auch auf Qwen3.8-27B. Hauptsetting 2 bit pro Gewicht plus FP16-Skala je 128 Spalten. Qwen3-8B,
  mittlere PPL: BF16 8.93, QTIP 11.10, LLVQ 10.86, CoS-VQ 10.56. Ablation: gemeinsame Index- und Skalensuche
  bringt mehr als reines Skalen-Feintuning.
- **NestQuant** (arXiv 2502.09720): verschachtelte Gosset-Gitter für W4A4KV4-Matrixprodukte; Ziel sind
  Aktivierungen, nicht reine Gewichtsquantisierung.
- **EntroLLM** (2505.02380), **ZipServ** (2603.17435): Entropie- bzw. verlustfreie Codierung. Hilft skalaren
  Codes, deren Entropie unter der Bitbreite liegt; Trellis-Pfadbits sind praktisch inkompressibel.
- **MCWC** (2605.24754): Vorhersage über Schichten plus gelerntes Entropiemodell; Gewichte müssen vor der
  Rechnung rekonstruiert werden, nicht im GEMM fusionierbar.

Einordnung (eigene Abschätzung, nicht gemessen): Ein Trellis mit 2^16 Zuständen liegt grob 0.4 bis 0.5 dB über
der Gauß-Schranke (QTIP-Werte bei 2 bit). E8 mit Entropiecodierung liegt etwa 0.65 dB darüber, Leech etwa 0.5 dB.
Den ganzen Abstand zu schließen wären rund 0.075 bit, bei 3.3 bis 3.7-facher KL pro Bit höchstens etwa 10 % KL.
Ergebnis: kein besseres Format bei 4 bis 4.5 bpw belegt. Übertragbar ist die gemeinsame Index- und Skalensuche
von CoS-VQ auf die Zeilenskala von IQR-V3, die heute vor GPTQ fest gewählt wird.
