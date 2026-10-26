# IT-Liquidation und Resale-Auktionen (DE/EU), Stand Oktober 2026

## Research status and access limits

### Takeaway
Im Recherche-Zeitraum fand sich kein öffentlich sichtbarer, aktueller (Oktober 2026) Auktionskatalog mit konkreten GPU-Servern oder NVIDIA-Datacenter-GPUs von deutschen oder EU-Liquidatoren. Die Suchergebnisse lieferten Marktkontext, Ankaufsanbieter und Preishinweise, aber keine verifizierbaren Lots mit Gebot, Enddatum und Verkäufertyp. Die direkte Prüfung von Plattformseiten war nicht möglich, weil die Sandbox-Proxy-Richtlinie die Verbindung verweigerte (siehe unten).

### Cited Findings
- Die Live-Abfrage von eBay.de, altatechnologies.com, hetzner.com, idealo.de, geizhals.de, bca-europa.de, auktionshaus-ruhr.de, buysellram.com, itscope.com und wirelesslogic.de ist am 2026-10-26 fehlgeschlagen: Der Egress-Proxy beantwortete CONNECT mit 403 (curl exit 56 bzw. kein Verbindungsaufbau). Diese Seiten sind daher nicht geprüft; Lots und Preise stammen ausschließlich aus Suchschnipseln.
- Ein eBay.de-Angebot "Gigabyte G291-281 2HE GPU Server, 2x Xeon Gold 6154, 768 GB RAM, 8 Bay, 8x GPU PCIe Slot", von einem gewerblichen Verkäufer, wurde in Suchergebnissen mit ca. EUR 2.730,48 (umgerechnet aus GBP ca. 2.400) angezeigt — https://www.ebay.de/itm/406136640482 (Snippet, Stand Suche 2026-10-26; Enddatum, Gebotsstand und Zustand nicht verifiziert). Ein zweites Gigabyte-G291-281-Angebot existiert unter https://www.ebay.de/itm/204648783203 (Preis und Datum nicht im Snippet).
- Gebrauchte Dell-Precision-Workstations mit RTX-Karten (eBay, Festpreis oder unklar): Dell Precision 7560 mit RTX A5000 ca. USD 1.199; Precision 7560 mit RTX A4000 USD 549,99; Precision 7550 mit Quadro RTX 3000 ca. EUR 585 (kein US-Versand) — Quellen: https://www.ebay.de/itm/127177494751 u. a. (Listen-Snippets; nicht als Auktion verifiziert). Einzelkarten: Precision 7670 mit RTX A1000 USD 129,99 und Precision 7560 mit RTX A2000 USD 139,99 bei einem US-Händler (https://gotlaptopparts.com/products/dell-precision-7670-nvidia-quadro-rtx-a1000-video-card-qn20-p1-r-a1-307009843438). Diese Preise sind Angebotspreise, keine Zuschläge.
- Alta Technologies (US-Händler) führt gebrauchte NVIDIA A100-, H100- und H200-Karten sowie komplette GPU-Server als Festpreisangebote (laut Suchsnippet; Website nicht abrufbar): https://altatechnologies.com/collections/used-nvidia-gpus
- Ankaufs-/Resale-Dienstleister mit Serverfokus (Werbeseiten, keine Auktion): Second-IT (DSGVO-konforme Löschung), Flagship Technologies (https://store.flagshiptech.com/Sell-to-Us), LADS (https://lads.com/?p=30). Keine Lots gefunden.
- Insolvenz-/Liquidationskanal: Die Suche fand keine Verwertungsplattform für Serverhardware eines deutschen Insolvenzverwalters. Ein Forumsbeitrag (administrator.de) nennt Abschreibung 3 Jahre und Preisbildung über Angebot/Nachfrage; das ist Forumsmeinung, keine Rechtsquelle: https://administrator.de/forum/restlicher-wert-von-hard-und-software-264391.html
- Leasing-Rückläufer: Die gefundenen Auktionsplattformen (CarOnSale/Mercedes-Benz, BCA/Alphabet, Arval MotorTrade, Allane/Auto1) betreffen ausschließlich Pkw; für IT-Hardware nicht relevant. IBM beschreibt Leasing-Vertragsende mit Kauf zum Restwert, nennt aber keine Online-Auktion: https://ibm.com/financing/end-of-lease
- Rückkauf statt Auktion: Eine Evernex-Fallstudie beschreibt einen deutschen Bankkunden, der über 500 ungenutzte Dell- und Oracle-Server im Rahmen eines Buy-back-Programms verkaufte (Zeitdruck wegen Platzbedarf): https://evernex.com/casestudy/evernex-buy-back-program-for-unused-servers/ (Herstellerangabe, keine Preise).
- Tradefinity (ITAD, Frankfurt/Hamburg) wurde 2022 von CSI Lifecycle Leasing GmbH übernommen: https://www.csileasing.com/news/csi-lifecycle-leasing-gmbh-acquires-top-itad-company-in-germany/ — Stand 2022, daher für Oktober 2026 als möglicherweise veraltet zu markieren; aktueller Verwertungsstatus nicht geprüft.
- Hetzner: Die Hetzner-"Serverbörse" ist ein Mietserver-Marktplatz für Dedicated Server und keine Liquidationsauktion für gebrauchte Hardware (Suchsnippet; Seite nicht abrufbar). GPU-Mietserver: Hetzner kündigte laut Suchsnippet die GEX45 (Sep. 2026) und GEX63 mit NVIDIA RTX PRO 6000 Blackwell Max-Q, 96 GB GDDR7 ECC (Okt. 2026) an: https://www.hetzner.com/de?country=us (Neuware/Miete, nicht Gebrauchtmarkt).
- Gebrauchtpreise H100 80GB (Sekundärmarkt, Aggregatoren, nicht verifizierte Transaktionen):
  - amcompute: typischer realisierter Preis H100 SXM 80GB ca. USD 14.250 (Aug. 2026), 25. Perzentil ca. USD 13.200; Stichprobe klein (https://www.amcompute.com/gpu-server-values).
  - GPUnex: stabilisiert bei USD 18.000–22.000 (2026).
  - Compute Exchange: USD 15.000–28.000 (gebraucht, nicht refurbished); USD 21.000–34.000 (refurbished) (Snippet, Quelle siehe Suchergebnisse).
  - Ein Bericht nennt USD 6.000–22.000 Sekundärmarkt-Spanne Juli 2026, bei ursprünglich USD 40.000 (2023); Quelle nicht eindeutig.
  - Widerspruch: Die Quellen streuen um ca. USD 6.000 bis 28.000; die Spanne ist breit, Transaktionsdaten fehlen.
- Gebrauchtpreis A100 80GB: SMM (10. Okt. 2026) meldet ein Angebot von 2 A100-80G-Systemen zu 1,16 Mio. CNY pro Einheit (getestet, ohne Garantie); Einzelmeldung, kein Index: https://news.metal.com/newscontent/104149650-smm-computing-power-news-a100-80g-second-hand-machine-reported-again-at-116-million
- RTX 6000 Ada: Keine Gebrauchtpreise 2026 gefunden. Neupreise zum Launch laut Notebookcheck USD 7.377,71–8.209,65 (https://www.notebookcheck.net/NVIDIA-RTX-6000-Ada-Generation-Retailer-pricing-points-to-imminent-release-for-AD102-based-workstation-GPU.672143.0.html). Mietpreise laut Thundercompute (Mai 2026): USD 0,61–0,99/h (https://www.thundercompute.com/blog/nvidia-rtx-6000-ada-pricing).
- Marktlage: BuySellRam beschreibt volatile Sekundärmärkte für H100/H200/A100; neuere Blackwell-Systeme drücken Restwerte älterer Systeme (https://www.buysellram.com/blog/nvidia-a100-h100-h200-cluster-liquidation-maximize-roi-and-asset-recovery/). Herstellerwerbung, Preise nicht geprüft.

### Inferences
- Die verfügbaren Lots (eBay-Angebote) sind überwiegend Festpreis- oder unklare Format-Angebote von Händlern, nicht von Liquidatoren oder Insolvenzverwaltern. Eine Auktion mit Zuschlag ist nicht belegt.
- Für Workstations mit RTX A4000/A5000 liegen Angebotspreise im Snippet-Bereich; ohne Zugriff auf Verkaufsdaten ist keine belastbare Referenz für Zuschlagspreise möglich.
- Der H100-Preisbereich ist breit; der untere Bereich (USD 13–15k) bezieht sich laut Quelle auf PCIe/SXM ohne Form-Faktor-Trennung in mindestens einer Quelle.

### Gaps
- Keine aktuellen Auktionskataloge deutscher Resteverwerter (z. B. Plattformen für Insolvenz- oder Rechenzentrumsliquidation) gefunden; Suchergebnisse waren nicht themenspezifisch.
- Keine Pflichtenlage oder Plattform-Nachweis für Insolvenzverwertung (InsO § 159, § 166) aus Quellen belegt.
- Kein idealo- oder Geizhals-Preisverlauf für Gebrauchtkarten abrufbar (Seiten geblockt).
- Keine A100/H100/L40S/A40-Lots mit Gebot, Enddatum und Verkäufertyp in DE/EU verifiziert.
- Alle eBay.de-, Alta-, Hetzner-, Idealo- und Geizhals-Seiten: Zugriff verweigert (Proxy 403), daher nicht geprüft.

## Key question 1: Which DE/EU IT liquidation or resale platforms are active in 2026?

### Takeaway
Keine aktive, belegte DE/EU-Auktionsplattform für Serverhardware mit Stand Oktober 2026 wurde gefunden. Sichtbar sind eBay.de-Händler (Restposten-Kategorie), US-Händler mit EU-Versand und Ankaufsdienstleister. Die Auktionsbelege beziehen sich auf Autos oder Leasing-Pkw.

### Cited Findings
- eBay.de-Sellers nutzen "RESTPOSTEN" für Massenware; gebrauchte Server und Workstation-Karten erscheinen dort ebenfalls: https://www.ebay.de/str/cprrefurbishedcomputers (Storefront-Snippet, Verkäufertyp nicht geprüft).
- Alta Technologies verkauft Karten und Server (Festpreis laut Snippet): https://altatechnologies.com/collections/used-nvidia-gpus
- Forumsnutzer zweifeln an der Seriosität von B2B-Restposten-Plattformen; keine Antwort im Thread: https://administrator.de/forum/grosshaendler-154497.html (Forumsmeinung).
- Ein 1999er Artikel zur Auktionsplattform auctionline.ch (COS) ist veraltet und nur historisch relevant: https://www.itreseller.ch/Artikel/1117/Auktions-Website_von_COS.html

### Inferences
- Der Markt für Serverhardware-Liquidation in DE scheint über Ankäufer, ITAD-Dienstleister und eBay-Händler zu laufen, nicht über öffentliche Auktionsportale. Das ist aus Suchsnippets abgeleitet und nicht durch Plattformabfragen bestätigt.

### Gaps
- Kein Nachweis aktiver Auktionen von Insolvenzverwaltern oder Rechenzentrumsbetreibern zu GPU-Servern.
- Die Suche nach "Bergungsauktion" und Auktionshäusern für IT (z. B. Versteigerer) lieferte keine relevanten Treffer.

## Key question 2: Current lots of NVIDIA data-center GPUs or GPU servers at auction?

### Takeaway
Keine verifizierten, aktuellen (Okt. 2026) Auktionslots für A100, H100, L40S, A40 oder A6000 in DE/EU gefunden. Ein eBay.de-GPU-Server (Gigabyte G291-281) ist als Angebot sichtbar; Status unbekannt.

### Cited Findings
- Gigabyte G291-281 2HE GPU Server, 2x Xeon Gold 6154, 768 GB RAM, 8 Bay, 8x GPU-Slot, Angebot gewerblicher Verkäufer, ca. EUR 2.730 (GBP 2.400): https://www.ebay.de/itm/406136640482 (Snippet; Format und Enddatum nicht verifiziert).
- Gigabyte G291-281 (zweites Angebot): https://www.ebay.de/itm/204648783203 (keine Preisangabe im Snippet).
- SMM: 2 A100-80G-Systeme zu 1,16 Mio. CNY/Stück (chinesischer Markt, nicht EU): https://news.metal.com/newscontent/104149650-smm-computing-power-news-a100-80g-second-hand-machine-reported-again-at-116-million

### Inferences
- Das eBay-Gigabyte-Angebot ist ein Beispiel für einen GPU-Server in Gebrauchtverkauf; ohne Enddatum oder Gebot ist er nur als Festpreis- oder Angebotsindikator zu lesen.

### Gaps
- Keine Lots mit NVIDIA-Datacenter-GPU in DE/EU mit Gebot oder Enddatum.
- Keine Dell PowerEdge-, HPE ProLiant- oder Supermicro-GPU-Server-Lots mit verifiziertem Status.

## Key question 3: Workstations (Dell Precision, HP Z, Lenovo ThinkStation) with RTX Ada/Blackwell listed, and prices

### Takeaway
Gebrauchte Dell-Precision-Workstations mit RTX-A-Serie (A1000–A5000) sind auf eBay.de und US-Händlern gelistet; RTX-Ada- und Blackwell-Workstations wurden in den Suchergebnissen nicht als Gebraucht-Lots gefunden.

### Cited Findings
- Dell Precision 7560 mit RTX A5000: ca. USD 1.199 (eBay.de-Listing, Snippet): https://www.ebay.de/itm/127177494751 (u. a.; Auktion vs. Sofortkauf nicht erkennbar).
- Dell Precision 7560 mit RTX A4000: USD 549,99 (Snippet).
- Dell Precision 7550 mit Quadro RTX 3000: ca. EUR 585, kein US-Versand (Snippet).
- Dell Precision 7540 mit Quadro RTX 3000: als beendet gelistet, GBP 365 (Snippet; Enddatum nicht genannt).
- Einzelkarten: Precision 7670 RTX A1000 USD 129,99; Precision 7560 RTX A2000 USD 139,99 (gotlaptopparts.com, Ersatzteile, getestet laut Angebot): https://www.gotlaptopparts.com/products/dell-precision-7560-nvidia-rtx-a2000-4gb-video-card-gddr6-0g48jy-qn20-p3-a1-227275753186
- Ada-Referenz: RTX 6000 Ada Launch-Neupreis USD 7.377,71–8.209,65 (Notebookcheck, Neuware).

### Inferences
- Die Gebrauchtpreise für A5000/A4000-Workstations liegen im Snippet bei rund USD 550–1.200; Ada- und Blackwell-Workstations scheinen im Gebrauchtmarkt seltener gelistet, Grund: unklar.

### Gaps
- Keine HP Z8/Z6/Z4- oder Lenovo ThinkStation-P-Lots mit RTX-Ada/Blackwell-Karte als Gebraucht-Lot verifiziert.
- Keine Zuschlagspreise oder Auktionsergebnisse für Workstations.

## Key question 4: Reference prices for used market

### Takeaway
Gebrauchtpreise für H100 80GB liegen laut Aggregatoren zwischen ca. USD 13.000 und 28.000 (2026), stark streuend und nicht durch Transaktionsdaten gestützt. Für A100 und RTX-6000-Ada fehlen belastbare Gebrauchtpreise aus DE/EU.

### Cited Findings
- H100 80GB SXM, typisch realisiert ca. USD 14.250 (25. Perz. USD 13.200), Stand Aug. 2026: https://www.amcompute.com/gpu-server-values
- H100 gebraucht USD 18.000–22.000 (GPUnex, 2026): https://cyfuture.ai/kb/gpu/nvidia-h100-gpu-price-2026-buying-options (Quelle nicht eindeutig zu GPUnex; Snippet).
- H100 gebraucht USD 15.000–28.000; refurbished USD 21.000–34.000 (Compute Exchange, Snippet): https://gpuperhour.com/blog/nvidia-h100-price (Zuordnung nicht eindeutig).
- Neu-H100 ca. USD 25.000–40.000 je nach Formfaktor (laut Aggregator): https://cloudzero.com/blog/h100-gpu-cost/
- A100 80G Einzelangebot 1,16 Mio. CNY (SMM, 10.10.2026).
- RTX 6000 Ada: Neupreis USD 7.377–8.209 (Launch); Miete USD 0,61–0,99/h (Thundercompute, Mai 2026).
- Hinweis: Mehrere Quellen sind Anbieter- oder Marketingseiten; Preisangaben sind nicht als Transaktionsindex belegt.

### Inferences
- Ein sinnvoller Referenzkorridor für gebrauchte H100 80GB in 2026 ist aus den Quellen ableitbar auf ca. USD 14.000–22.000, mit PCIe-Karten eher am unteren Ende (Aggregator-Angabe); Käufer sollten Live-Angebote gegenprüfen.
- Gebrauchte Workstation-Karten (A4000/A5000) liegen laut Snippets in einem deutlich niedrigeren Preisbereich als Datacenter-GPUs.

### Gaps
- Keine idealo-, Geizhals- oder eBay-sold-Daten (geblockt).
- Keine Preisangaben in EUR für H100/A100 aus DE-Händlern verifiziert.
- Keine Gebrauchtpreise für RTX 6000 Ada.

## Limits and stale-data flags
- Die Live-Abruf-Versuche von Plattformseiten schlugen fehl (Proxy 403); alle Lot-Angaben stammen aus Suchsnippets und sind nicht gegen die Seite geprüft.
- Stale: Tradefinity-Übernahme (2022); auctionline.ch-Artikel (1999); Evernex-Fallstudie (Datum unbekannt).
- Mehrheit der Suchergebnisse für "Auktion" betraf Pkw-Leasing; nicht relevant.
