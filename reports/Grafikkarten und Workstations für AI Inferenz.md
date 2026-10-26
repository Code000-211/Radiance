# Auktionsportale bleiben für GPU-Käufer weitgehend unlesbar

Auf Insolvenz- und Zwangsversteigerungsplattformen ließ sich für Oktober 2026 **kein einziges Lot mit GPU oder Workstation für AI-Inferenz verifizieren**. Die Recherche-Umgebung konnte zvg-portal.de, zoll-auktion.de, auktion.bund.de, justiz-auktion.de, insolvenzbekanntmachungen.de, eBay.de, Catawiki, Troostwijk und auction.de nicht lesen (DNS-Fehler bzw. Proxy-Verweigerung). "Keine Lose gefunden" bedeutet deshalb ausdrücklich nicht "keine Lose vorhanden". Aus Suchsnippets liegen einige Hinweise vor: ein eBay.de-GPU-Server (Gigabyte G291-281) für rund 2.730 €, mehrere gebrauchte Dell-Precision-Workstations mit RTX-A-Karten zu Festpreisen, und zwei Intergavel-Workstations ohne GPU-Angabe. Die meisten dieser Hinweise sind undatiert, Festpreisangebote oder bereits abgelaufen, und kein Zuschlagspreis ist belegt. Belastbarer als die Lose sind die Preisreferenzen: Eine gebrauchte RTX 4090 liegt laut einer Sekundärauswertung bei median 2.104 € (August 2026), eine neue RTX PRO 6000 (96 GB) bei rund 15.400 bis 15.500 €, und gebrauchte H100 80 GB werden zwischen ca. 13.000 und 22.000 USD gehandelt. Wer bieten will, muss die Plattformen selbst öffnen und jedes Lot gegen diese Korridore und die Checkliste unten prüfen.

## Was verifiziert ist: der Zugriff, nicht das Angebot

Verifiziert ist ausschließlich der Zugriffsstatus. Die Notizen dokumentieren für zoll-auktion.de, auktion.bund.de, justiz-auktion.de und insolvenzbekanntmachungen.de einen Fehler bei der Namensauflösung (DNS); für zvg-portal.de, eBay.de und Catawiki verweigerte der Egress-Proxy den Verbindungsaufbau mit HTTP 403. Laut Proxy-Dokumentation ist diese Sperre eine Richtlinienentscheidung und wird nicht umgangen. Auch Troostwijk, auction.de, Lot-Tiger, Hood.de, Maxauktion, Alta Technologies, Hetzner und Idealo/Geizhals waren nicht abrufbar. Im Rahmen dieses Berichts wurde keine dieser Seiten erneut aufgerufen; die Aussagen stützen sich auf die drei Recherchenotizen im Ordner `research_notes`. Damit liegen für keine Plattform ein Lot-Status, ein Gebot, ein Enddatum oder ein Standort aus erster Hand vor. Die Websuche fand zudem keine deutsche Insolvenz- oder Zwangsversteigerung mit RTX 3090, 4090, 5090, RTX A6000, RTX 6000 Ada, RTX PRO 6000, A100, L40/L40S, H100, V100 oder T4. Auch das ist nur ein Befund aus Suchtreffern, keine Prüfung der Portale.

## Snippet-Funde: Festpreise, Abgelaufenes und ein ungeklärter Server

Die folgenden Funde stammen aus Suchsnippets und sind nicht gegen die Angebotsseite geprüft. Keiner ist als aktuelles Gebotslos bestätigt.

| Fund | Format und Stand | Vorbehalt |
|---|---|---|
| Gigabyte G291-281 2HE GPU-Server, 2x Xeon Gold 6154, 768 GB RAM, 8 Bay, eBay.de ([Angebot](https://www.ebay.de/itm/406136640482)) | Gewerblicher Verkäufer, ca. 2.730 € (aus 2.400 GBP umgerechnet), kein Enddatum im Snippet | "8x GPU PCIe Slot" beschreibt Steckplätze, nicht bestückte Karten. Zweites Angebot: [ebay.de/itm/204648783203](https://www.ebay.de/itm/204648783203), ohne Preis im Snippet. |
| Dell Precision 7560 mit RTX A5000, eBay.de ([Angebot](https://www.ebay.de/itm/127177494751)) | ca. 1.199 USD, Festpreis oder Auktion unklar | Die 7560 ist ein Notebook. Die mobile A5000 hat nach Herstellerangaben 16 GB, nicht 24 GB. |
| Dell Precision 7560 mit RTX A4000, eBay.de (Snippet) | 549,99 USD | Mobile A4000 mit 8 GB laut Huutokaupat-Titeln, siehe unten. |
| Dell Precision 7550 mit Quadro RTX 3000, eBay.de (Snippet) | ca. 585 €, kein US-Versand | Mobile RTX 3000 mit 6 GB, für größere Modelle zu knapp. |
| Dell Precision 7670 mit RTX A1000 und 7560 mit RTX A2000, US-Ersatzteilhändler ([A1000](https://gotlaptopparts.com/products/dell-precision-7670-nvidia-quadro-rtx-a1000-video-card-qn20-p1-r-a1-307009843438), [A2000](https://www.gotlaptopparts.com/products/dell-precision-7560-nvidia-rtx-a2000-4gb-video-card-gddr6-0g48jy-qn20-p3-a1-227275753186)) | 129,99 USD bzw. 139,99 USD, Einzelkarten | Angebotspreise, keine Zuschläge. Für Inferenz kaum nutzbar (4 GB). |
| Huutokaupat (Finnland): Dell Precision 7560, RTX A4000 8 GB, 32 GB RAM ([Lot](https://huutokaupat.com/en/kohde/6452113/dell-precision-7560-intel-core-i7-11850h-tehokannettava-15-6-fhd-32-gt-1024-ssd)) | Startpreis 0 €, ohne Gebote, Ende laut Snippet 5. August 2026 | Abgelaufen. Gleiches Modell mit 64 GB RAM endete am 31. Juli 2026 ([Lot](https://huutokaupat.com/en/kohde/6440770/dell-precision-7560-intel-core-i7-11850h-tehokannettava-15-6-fhd-32-gt-1024-ssd)). |
| Intergavel (Ungarn), IT-Auktion: Dell Precision 7820 mit zwei Xeon Gold 6252 ([Lot 3](https://intergavel.com/en/auctions/it-equipment-auction-2/3)) und Precision 7820 ([Lot 9](https://intergavel.com/en/auctions/it-equipment-auction-2/9)) | Startpreise 3.000 € bzw. 600 € | GPU-Ausstattung nicht angegeben. Enddatum nicht im Snippet. Der 7820 ist ein Tower; ob er eine Datacenter- oder RTX-Karte trägt, ist offen. |
| RTX 3090 (24 GB), eBay.de ([Angebot](https://www.ebay.de/itm/187746102862), [Angebot](https://www.ebay.de/p/17041884511)) | 349 € (ASUS ROG Strix, Neuware-Format unklar) bis 1.039 € (gebraucht, mit Käuferschutz) | Undatiert. Das 349-€-Angebot ist ein Ausreißer; Zustand unbekannt. |
| RTX 4090 AORUS Xtreme, Privatanzeige ([SweClockers](https://www.sweclockers.com/marknad/263654)) | 8.500 SEK nach Reparatur | Festpreis, kein Auktionslos, Datum unbekannt. |
| A100 80 GB PCIe, eBay.de aus Japan ([Angebot](https://www.ebay.de/itm/406086417179)) | ca. 15.700 € + ca. 255 € Versand | Kein Rückgaberecht, Einfuhrabgaben möglich, Datum unbekannt. Einzelkarte. |
| Dual-A100-80GB-Server, eBay.de UK ([Angebot](https://www.ebay.de/itm/277051254341)) | umgerechnet ca. 26.200 € | Festpreis-Format, Datum unbekannt. |
| A100-80G-Systeme, SMM-Meldung vom 10. Oktober 2026 ([Meldung](https://news.metal.com/newscontent/104149650-smm-computing-power-news-a100-80g-second-hand-machine-reported-again-at-116-million)) | 1,16 Mio. CNY "pro Einheit" | Chinesischer Markt. Die Angabe bezieht sich vermutlich auf ein ganzes System, nicht auf eine Karte, und ist ohne Systemangabe nicht mit Einzelkarten vergleichbar. |
| Alta Technologies (USA), gebrauchte A100, H100, H200 und GPU-Server ([Sortiment](https://altatechnologies.com/collections/used-nvidia-gpus)) | Festpreise laut Suchsnippet | Website nicht abrufbar. Versand und Zoll nach Deutschland nicht geprüft. |
| RTX 4090, defekt, Marktplaats NL ([Angebot](https://www.marktplaats.nl/v/computers-en-software/overige-computers-en-software/a1528964106-videokaart-gigabyte-geforce-rtx-4090-defect)) | Juli 2026 laut Zusammenfassung | Ausland, defekt, nur als Randnotiz. |

Weitere Treffer waren veraltet oder themenfremd: Eine EVGA-Prototypen-Versteigerung von 2022, ein Mindfactory-Bericht von März 2025 über RTX-40/50-Lieferstopps ohne bestätigte Auktion, und ein Artesian-Builds-Konkurs in den USA mit Lots ab 5.000 USD. Die Suche nach Liquidatoren für Serverhardware in Deutschland fand Ankäufer und ITAD-Dienstleister, aber keine aktive Auktionsplattform. Die Übernahme von Tradefinity durch CSI Lifecycle Leasing stammt aus dem Jahr 2022 und sagt über den heutigen Verwertungsstatus nichts aus.

## Referenzpreise: belastbarer als die Lose, aber uneinheitlich

Die stärkste Preisreferenz ist die Gebrauchtpreisauswertung für die RTX 4090: Laut [tech-insider.org](https://tech-insider.org/rtx-4090-vs-rtx-5080-gebrauchtmarkt-2026/) lag der **Median bei 2.104 €**, die mittleren 50 % zwischen 2.031 und 2.189 €, gemessen über PCPrice.Watch im August 2026. Das ist eine Sekundärquelle, die Kleinanzeigen- und eBay-Preise mischt und nicht unabhängig geprüft wurde. Umgerechnet ergibt das rund 88 € pro GB VRAM. Für die RTX PRO 6000 Blackwell (96 GB) gibt es nur Neupreise aus Preisvergleichen, keine Gebrauchtpreise: Der [Hardwaredealz-Vergleich](https://www.hardwaredealz.com/hardware/gpu/nvidia-rtx-pro-6000-blackwell-workstation-edition-96gb/nvidia-96gb-gddr7-4x-dp-retail/preisvergleich.27857) nennt Stand 3. Oktober 2026 einen niedrigsten Preis von 15.490 € und einen 12-Monats-Median von 12.192 €, [GPUPrix](https://gpuprix.com/eu/gpus/rtx-pro-6000-blackwell) nennt 15.390 € als Tief (19. September 2026). Einzelhändler wie [JACOB](https://www.jacob.de/produkte/nvidia-quadro-rtx-900-5g144-2500-000-artnr-100670590.html) listen 11.300 bis 11.600 € inklusive Mehrwertsteuer bei geringer Stückzahl, ohne Datum. Das ergibt je nach Quelle etwa 118 bis 161 € pro GB VRAM für neue Karten.

Für die RTX 6000 Ada (48 GB) fehlt ein EU-Gebrauchtpreis. Der US-Listenpreis lag bei 9.999 USD ([Tom's Hardware](https://www.tomshardware.com/news/rtx-6000-now-on-sale-8200)), der Launch-Korridor bei 7.377 bis 8.210 USD ([Notebookcheck](https://www.notebookcheck.net/NVIDIA-RTX-6000-Ada-Generation-Retailer-pricing-points-to-imminent-release-for-AD102-based-workstation-GPU.672143.0.html)). Als Mietpreis nennt [Thundercompute](https://www.thundercompute.com/blog/nvidia-rtx-6000-ada-pricing) im Mai 2026 0,61 bis 0,99 USD pro Stunde. Für die A100 80 GB PCIe nennt [JarvisLabs](https://jarvislabs.ai/ai-faqs/nvidia-a100-gpu-price) eine Neupreis-Orientierung von 9.500 bis 14.000 USD aus dem Jahr 2025. Der Gebrauchtpreis ist nur ein Einzelangebot aus Japan bei rund 15.700 €.

Für die H100 80 GB streuen die Aggregatoren stark. [amcompute](https://www.amcompute.com/gpu-server-values) nennt im August 2026 einen typischen realisierten Preis von rund 14.250 USD und ein 25-Perzentil von 13.200 USD, die Stichprobe ist aber klein. [Cyfuture](https://cyfuture.ai/kb/gpu/nvidia-h100-gpu-price-2026-buying-options) nennt 18.000 bis 22.000 USD, [GPUPerHour](https://gpuperhour.com/blog/nvidia-h100-price) 15.000 bis 28.000 USD gebraucht und 21.000 bis 34.000 USD refurbished. Die Zuordnung der Preise zu Anbietern ist in beiden Quellen unklar, und keine trennt sauber zwischen SXM- und PCIe-Form. Neu liegt die H100 laut [CloudZero](https://cloudzero.com/blog/h100-gpu-cost/) bei 25.000 bis 40.000 USD. Der sinnvolle Gebrauchtkorridor liegt daher grob bei 13.000 bis 22.000 USD, und die Form (SXM oder PCIe) entscheidet über den Preis innerhalb dieses Korridors.

## Konkrete Schritte für die manuelle Suche

Da keine Plattform aus der Recherche-Umgebung lesbar war, bleibt die Prüfung Handarbeit. Die folgende Tabelle nennt die Plattformen mit Suchbegriffen und Filtern. Die Erwartungen beruhen auf allgemeiner Kenntnis der Plattformen und sind nicht in den Notizen belegt.

| Plattform | Suchbegriffe | Filter | Erwartung |
|---|---|---|---|
| zvg-portal.de | Server, Grafikkarte, Workstation, Rechenzentrum | Bewegliche Sachen, falls vorhanden; nach Versteigerungstermin sortieren | Überwiegend Immobilien; IT-Lose sind selten. |
| zoll-auktion.de | GPU, Grafikkarte, Server, Workstation | Kategorie IT/Elektronik, Standort Deutschland | Zollgut mit eingeschränkter Gewährleistung. Bedingungen lesen. |
| auktion.bund.de | Server, Workstation, IT-Hardware | Kategorie IT, Enddatum | Bundesverwertung, IT gelegentlich. |
| justiz-auktion.de | Server, Grafikkarte, IT | Bundesland, Enddatum | Insolvenz- und Vollstreckungsgut. |
| insolvenzbekanntmachungen.de | Rechenzentrum, Serverhersteller, Systemhaus, IT | Verfahrensart, Zeitraum | Kein Auktionsportal. Verfahren finden, dann Insolvenzverwalter direkt anfragen. |
| eBay.de | "A100", "H100", "RTX A6000", "L40S", "Tesla", "GPU Server" | Auktion, Standort Deutschland, Zustand gebraucht; "beendete Angebote" für Preise | Häufig Festpreis-Händler; Auktionen mit Enddatum gezielt filtern. |
| Catawiki | GPU, Grafikkarte, Workstation, Server | Kategorie Computer, laufende Auktionen | Eher Sammlerstücke. Käuferaufgeld in den Bedingungen prüfen. |
| Troostwijk | IT, Server, GPU | Standort Deutschland, Kategorie IT | Liquidator mit großer Lot-Zahl. |
| auction.de | IT, Server, Workstation | Kategorie IT | Vollständigkeit unbekannt. |

Jedes Lot sollte vor einem Gebot anhand der folgenden Punkte geprüft werden:

| Prüfpunkt | Warum | Wie prüfen |
|---|---|---|
| Exakte Modellbezeichnung | A100 PCIe und SXM4, H100 PCIe und SXM5, RTX 4090 Founders und Partnerkarten unterscheiden sich in Preis und Anschluss. | Aufkleber, Platinenfoto, Seriennummer, nach dem Kauf `nvidia-smi`. |
| VRAM | Entscheidet, welche Modelle laufen. | Mobile und Desktop-Varianten unterscheiden sich. Bei Notebooks Modellnummer mit Herstellerdatenblatt abgleichen. |
| Bestückt oder nur Chassis | Eine Angabe wie "8x GPU-Slot" heißt nicht, dass Karten dabei sind. | Lot-Beschreibung und Fotos der Steckplätze. |
| Kühlung | Datacenter-Karten (A100, H100, L40S, A40) haben meist keinen Lüfter und brauchen Serverluftstrom. | Gehäuse und Lüfterkonzept mit dem geplanten Aufbau abgleichen. |
| Stromanschluss | Die RTX 4090 nutzt 12VHPWR, Datacenter-Karten oft EPS oder Serverstromversorgung. | Netzteil- und Mainboard-Datenblatt. |
| Zustand und Funktion | Gebrauchte Karten können durch Dauerlast verschlissen sein. | Benchmark-Lauf, Temperaturen, bei Datacenter-Karten ECC-Fehlerzähler. |
| Herkunft und Verkäufer | Insolvenzverwalter, Privatperson und Händler haben unterschiedliche Garantie- und Rückgabelagen. | Impressum, Auktionsbedingungen. |
| Gewährleistung und Rückgabe | Versteigerungen schließen Gewährleistung oft aus. | Bedingungen auf Ausschluss prüfen. |
| Gesamtkosten | Gebot, Aufgeld, Versand, Einfuhrabgaben und Abholung summieren sich. | Gesamtpreis vor dem Gebot berechnen. |
| Enddatum und Gebotsstand | Abgelaufene Lose sind nicht kaufbar; ein Preis ohne Gebote ist kein Marktpreis. | Status "beendet" oder "läuft" prüfen. |
| Preisvergleich | Neu- und Gebrauchtpreise nicht mischen. | Gegen die Korridore in der Referenztabelle prüfen. |

Für die VRAM-Frage gilt als Faustregel (keine Quelle aus der Recherche): Die Modellgewichte brauchen ungefähr Parameter in Milliarden mal Bits pro Gewicht geteilt durch 8, dazu Puffer für den KV-Cache. Ein 70B-Modell mit 4 Bit passt daher grob in 48 GB, aber nicht in 24 GB. Die 96 GB der RTX PRO 6000 öffnen größere Modelle, kosten aber ein Vielfaches einer 4090.

## Conclusion

Die eigentliche Antwort lautet nicht "nichts vorhanden", sondern "nicht prüfbar aus dieser Umgebung". Die Lücke ist strukturell: Datacenter-GPUs und Server aus Insolvenzen laufen laut Notizen selten über öffentliche Auktionsportale und häufiger über direkte Verwertung durch Insolvenzverwalter oder Ankäufer. Für einen Käufer heißt das, dass die Anfrage beim Insolvenzverwalter nach IT-Bestand oft ergiebiger ist als eine weitere Portalsuche. Der nächste Schritt ist eine manuelle Browser-Prüfung der genannten Plattformen, nicht eine weitere automatisierte Suche.

Die Preisdaten zeigen zugleich, wo ein Schnäppchen wahrscheinlich nicht liegt. Eine 96-GB-Karte, die deutlich unter dem Neupreis-Tief von rund 15.400 € angeboten wird, ist entweder ein echter Gebrauchtfund oder ein Warnsignal, und ohne Zustandsnachweis und Herkunft gilt das Warnsignal. Die Notizen enthalten außerdem drei Widersprüche, die vor jeder Entscheidung zu klären sind: Die A100-Meldung von SMM bezieht sich vermutlich auf ein System, die RTX-PRO-6000-Preise sind Neuware, und die H100-Quellen mischen SXM und PCIe.
