# Insolvenz- und Zwangsversteigerungsplattformen: GPUs und Workstations für AI-Inferenz (Stand Oktober 2026)

Status: Die Recherche konnte keine Lots auf den offiziellen Plattformen prüfen. Die Zielseiten sind aus der Recherche-Umgebung nicht erreichbar. Es gibt keine belastbaren Lot-Daten, keine Preise und keine Enddaten aus Oktober 2026.

## Zugriffsstatus der Plattformen

### Takeaway
Keine der geforderten Plattformen konnte gelesen werden. Der Egress-Proxy der Umgebung verweigert die Verbindung zu diesen Hosts mit HTTP 403 (Richtlinie). Die Fehlermeldung kam vom Proxy und nicht vom Zielserver. Nach der Proxy-Dokumentation wird eine solche Verweigerung nicht umgangen und nicht wiederholt.

### Cited Findings
- zoll-auktion.de: Host nicht auflösbar (DNS `ENOTFOUND` über WebFetch); curl ohne Antwort (HTTP 000).
- auktion.bund.de: Host nicht auflösbar; curl ohne Antwort.
- justiz-auktion.de: Host nicht auflösbar; curl ohne Antwort.
- insolvenzbekanntmachungen.de: Host nicht auflösbar; curl ohne Antwort.
- zvg-portal.de: curl `CONNECT tunnel failed, response 403` (Proxy-Richtlinie).
- Proxy-Status (`/__agentproxy/status`) verzeichnet `connect_rejected` mit 403 für Hosts wie www.ebay.de, www.catawiki.com und www.heise.de. Das heißt: Die Umgebung blockiert allgemein Auktions- und Nachrichtenhosts, nicht nur die Insolvenzportale. Ursache: Egress-Policy der Session (siehe Proxy-README unter /root/.ccr/README.md).

### Inferences
- Ein Zugriff von dieser Session aus ist für die Plattformen nicht möglich. Die Sperre liegt im Egress-Setup und nicht an den Seiten selbst. Eine Prüfung von einem Rechner mit Internetzugang ist nötig, bevor Lots bewertet werden können.

### Gaps
- Lot-Titel, Modell, VRAM, Zustand, Gebot oder Startpreis, Enddatum und Standort konnten für keine Plattform erhoben werden.
- Ob zvg-portal.de, justiz-auktion.de, zoll-auktion.de oder auktion.bund.de aktuell GPU- oder Server-Lots führen, ist unbekannt.
- Die Frage nach Lots aus Rechenzentrums- oder Systemhaus-Insolvenzen konnte nicht beantwortet werden.

## Ergebnisse der Websuche (Stichproben, nicht verifiziert)

### Takeaway
Die Websuche fand keinen aktuellen deutschen Insolvenz- oder Zwangsversteigerungs-Lot für GPUs, Workstations oder Server. Die Treffer sind veraltet, betreffen andere Länder oder sind nicht einschlägig.

### Cited Findings
- Suchtreffer "Grafikkarte RTX 4090 Insolvenz Auktion": Eine defekte Gigabyte RTX 4090 wurde in einer niederländischen Faillissementsveiling angeboten. Das Inserat soll laut Zusammenfassung aus Juli 2026 stammen. Die Angabe konnte nicht geprüft werden, die Seite war nicht abrufbar. Die Quelle ist eine Marktplaats-Seite (https://www.marktplaats.nl/v/computers-en-software/overige-computers-en-software/a1528964106-videokaart-gigabyte-geforce-rtx-4090-defect). Sie liegt außerhalb des Scopes (NL) und ist nur als Hinweis zu werten.
- EVGA-Prototyp-Versteigerung: Ein Auktionsangebot aus 2022, kein Insolvenzfall (https://extreme.pcgameshardware.de/threads/627828). Veraltet.
- Mindfactory (März 2025): Berichte, dass RTX 40/50 aus dem Verkauf genommen wurden, vor möglicher Insolvenz (https://www.profesionalreview.com/2025/03/06/mindfactory-ha-retirado-todas-las-tarjetas-rtx-40-50-por-un-posible-problema-de-insolvencia/). Keine bestätigte Auktion gefunden. Veraltet (2025).
- Artesian Builds (USA): Insolvenzauktion mit GPUs in Lots ab 5.000 US-Dollar Schritt (https://www.notebookcheck.net/Artesian-Builds-puts-thousands-of-PC-parts-graphics-cards-and-partly-built-systems-worth-nearly-US-1-million-up-for-bankruptcy-auction.630388.0.html). Nicht Deutschland, und Lots dieser Größe sind für Einzelkäufer nicht geeignet.
- Datacenter Almere (Niederlande): Insolvenz eines Colocation-Anbieters mit Insolvenzverwalter; Auktion von Servern nicht belegt, Datum fehlt (https://datacenterdynamics.com/en/news/dutch-colo-firm-datacenter-almere-files-for-bankruptcy).
- Kaufring AG: Artikel zu einem Rechenzentrum in Düsseldorf im Insolvenzkontext; deutlich älter, kein Bezug zu 2026 (https://www.textilwirtschaft.de/fashion-management/ausverkauf-beim-kaufring-36413). Die Quelle wirkt themenfremd und wurde nur als Hinweis gewertet.

### Inferences
- Der Markt für gebrauchte Consumer-GPUs und Server-Hardware wird über Insolvenzen eher selten im öffentlichen Auktionsangebot sichtbar. Belastbare Aussagen brauchen direkten Zugriff auf die Plattformen.

### Gaps
- Keine deutsche Insolvenzauktion mit RTX 3090/4090/5090, RTX A6000, RTX 6000 Ada, RTX PRO 6000, A100, L40/L40S, H100, V100 oder T4 gefunden.
- Keine Serveranbieter- oder Rechenzentrums-Liquidation aus Deutschland mit Versteigerung 2026 gefunden.
- Kein Preisvergleich möglich.

## Empfohlene nächste Schritte

1. Die Plattformen von einem Rechner mit Internetzugang abfragen: zvg-portal.de, zoll-auktion.de, auktion.bund.de, justiz-auktion.de, insolvenzbekanntmachungen.de. Suchbegriffe: Grafikkarte, GPU, Workstation, Server, Rechenzentrum, Serverschrank.
2. Bei Insolvenzverwaltern direkt nachfragen (Verwertungsangebote für IT-Bestand), da solche Verkäufe oft nicht öffentlich ausgeschrieben werden.
3. Die Auktionen erst nach Abruf der Lot-Seiten mit Datum und Preis neu bewerten. Veraltete Treffer (2022, 2025) sind ausgeschlossen.

## Quellen
- Proxy-Statusdaten: `curl -sS http://127.0.0.1:44445/__agentproxy/status` (Einträge `connect_rejected`, 403).
- Proxy-Dokumentation: /root/.ccr/README.md.
