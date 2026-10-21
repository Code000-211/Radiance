# Vertrag: adaptives Baumfenster >3 (P146-Redesign, Stand 2026-10-18)

**Vorgeschichte:** Der erste Fallback-Versuch (Commits 3006c30/517aeeb, revertiert
als 54882d2/1f0c10f) verglich eine angenommene PFADLÄNGE mit einem KNOTEN-Budget
und verengte so selbst ein perfektes Baummodell. Die Lehren stehen hier als
Vertrag; eine Umsetzung startet erst nach Review dieses Dokuments.

**Review-Status: nicht implementierungsreif.** Das CPU-Review hat eine fehlende
Topologieinformation gefunden. `tree_path_depth` allein bestimmt kein Knotenbudget.
Die Signatur in Abschnitt „Der Vertrag“ bleibt ein Entwurf, keine Freigabe.

Ein Baum und eine Kette können beide Pfadtiefe 8 haben, aber unterschiedlich viele
Knoten bis Tiefe 4. Außerdem ersetzt `Engine::publish_block_drafts` einen Baum
bei QSA-Grenze oder verweigertem Branching durch seine Greedy-Kette
(`core/engine_draft.cpp:570-595`). Ein Faktor aus der konfigurierten Variante
beschreibt dann nicht den tatsächlich veröffentlichten Baum.

Vor Umsetzung muss der Vertrag daher festlegen:

- Umrechnung aus der **aktuellen veröffentlichten Topologie**, nicht allein aus
  maximaler Pfadtiefe oder Variantenname. Am Observe-Punkt ist die Topologie
  noch vorhanden; Retirement folgt danach (`core/sched/scheduler.cpp:1849-1864`).
- Vollannahme darf durch eine kleinere tatsächlich veröffentlichte Topologie
  oder eine vorübergehende Konkurrenzdecke kein größeres gespeichertes Fenster
  versehentlich verengen. Verengen und Befördern brauchen disjunkte Bedingungen.
- Ein tatsächlicher Nicht-Baum-Fall mit Pfadtiefe 0 darf nicht wegen
  `accepted_path >= 0` automatisch befördern.
- „Eine Knotenebene“ muss eindeutig als Knotenzahl oder Pfadebene definiert
  werden; diese Einheiten sind nicht austauschbar.
- Tests müssen verzweigten Baum und Kette mit gleicher maximaler Pfadtiefe,
  Topologiewechsel und Konkurrenzwechsel im geschlossenen Regelkreis prüfen.

Die endgültige Policy und die bindende Spezifikation müssen vor dem Cutover
übereinstimmen. Kein Budget->3-Benchmark auf Grundlage dieses ungeklärten Entwurfs.

## Einheiten (der Kernfehler des ersten Versuchs)

- `max_depth`/`budget` (Seitenwagen) ist ein **Knotenbudget**: die feste
  15-Knoten-Topologie (2 Knoten je Tiefe 1..7, 1 auf Tiefe 8) hat einen
  längsten PFAD von 8.
- `accepted` (observe) ist eine **Pfadlänge**: höchstens 8 bei dieser Topologie.
- **Regel:** Fenstereinheit = Knotenbudget; alle Vergleiche mit Annahmedaten
  laufen über die maximale Pfadtiefe `tree_path_depth`, die der Aufrufer kennt
  (die Topologie ist deklariert; Tiefe 8 gehört zur festen Variante).

## Der Vertrag

1. `observe(drafted_nodes, accepted_path, active_seqs, tree_path_depth)`:
   `tree_path_depth` ist die deklarierte maximale Pfadtiefe des geplanten
   Baums (0 für nicht-Baum-Beobachtungen). Nur mit dieser Größe wird
   verglichen.
2. Verengen: `if (accepted_path + 1 < window) window = accepted_path + 1`
   — in PFAD-Einheiten, dann auf Knotenbudget umgerechnet (der Umrechnungs-
   faktor kommt aus der deklarierten Topologie, nicht aus dem EMA).
3. Befördern: nur wenn `accepted_path >= tree_path_depth` (der Baum wird in
   voller Tiefe angenommen), wächst das Fenster um eine Knotenebene pro
   Schritt bis zur Decke.
4. Getter sind rein: `next_depth`/`current_depth` klemmen nur gegen die
   Konkurrenzdecke; **keine** bewegt Politik.
5. §4: adaptiv aus ⇒ statisch 3. adaptiv an ⇒ Decke = deklariertes Budget,
   verengt/befördert ausschließlich durch `observe`.
6. Die EMA-Annahmelänge bleibt ein berichtetes Diagnose-Maß (spec §14) und
   KEIN Policy-Eingang — der erste Versuch hat sie als Eingang benutzt und
   genau da die Einheiten verwechselt.

## Test-Pflichten vor irgendeiner Integration

- Geschlossener Regelkreis mit ehrlicher Spur: `d = next_depth();
  observe(d, a, ...)` mit `a` ≤ Pfadtiefe des entworfenen Baums; Kollaps auf
  Pfadtiefe 1 bei Nullannahme; Wiederaufstieg auf die Decke bei Vollannahme —
  alles gegen die Topologie, die der Test deklariert.
- Perfekter verzweigter Baum: 15 Knoten, Pfadtiefe 8; Vollannahme muss die
  Decke halten, NICHT auf 8 verengen.
- Konkurrenzübergänge, wiederholte Lesevorgänge, configure/reset setzen das
  Fenster zurück.
- `tree_sampling_path_test`: tiefe Fälle laufen mit adaptiv an (§4); die
  Verengungs-Erwartung folgt diesem Vertrag, nicht der alten EMA-Arithmetik.

## Messpflicht vor Aktivierung im Betrieb

Budget 3 gegen Budget 15 mit DIESEM Fallback, BetterBench je Kategorie — der
frühere 3-vs-15-Vergleich (88.5 vs 55.7 tok/s) lief mit dem revertierten
fehlerhaften Fallback und zählt nicht.