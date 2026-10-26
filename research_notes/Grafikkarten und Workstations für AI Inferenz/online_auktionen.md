# Online-Auktionen: GPUs und Workstations für AI-Inferenz (Deutschland/EU, Stand Oktober 2026)

Recherchedatum: 2026-10-26. Ergebnis vorab: Es konnte **kein einziges aktuelles (Okt. 2026) Auktionslos** für RTX 3090/4090/5090, RTX A6000, RTX 6000 Ada, RTX PRO 6000, A100, L40S oder H100 verifiziert werden. Die Auktionsplattformen selbst waren aus dieser Umgebung nicht erreichbar (siehe Gaps). Die Preisangaben unten stammen aus Suchergebnis-Snippets und sind größtenteils undatiert oder sekundär; sie sind als Referenz, nicht als Auktionsbeleg zu lesen.

## Zugriffsstatus der Plattformen

- eBay.de (Auktionsfilter `LH_Auction=1`): per curl `CONNECT tunnel failed, response 403` vom Egress-Proxy. Laut /root/.ccr/README.md ist 403 eine Richtlinienentscheidung für diesen Host; nicht umgangen.
- catawiki.com: per curl ebenfalls 403 (Tunnel verweigert). Nicht umgangen.
- troostwijkauctions.com, huutokaupat.com, auction.de, Lot-Tiger, Hood.de, Maxauktion, Mister Auction, Intergavel: WebFetch lieferte `getaddrinfo ENOTFOUND` (Host nicht auflösbar). Nur huutokaupat.com/intergavel.com wurden über die Websuche indexiert, nicht direkt gelesen.
- Keine Anmeldung, keine Registrierung, keine Gebote.

## Frage 1: Welche Plattformen listen relevante Ware?

### Takeaway
Über die Websuche ließ sich nur bestätigen, dass Finnische und Ungarische IT-Auktionshäuser (Huutokaupat.com, Intergavel) Dell-Precision-Workstations mit Quadro-/RTX-GPUs führen, überwiegend bereits abgelaufen. Für eBay.de und Catawiki ergaben Suchtreffer nur Festpreis- oder Altangebote. Lot-Tiger und Lot-Tiger-ähnliche Liquidatoren konnten nicht gefunden werden.

### Cited Findings
- Huutokaupat (Teraset Finland Oy, Helsinki): Dell Precision 7560, Core i7-11850H, 32 GB RAM, 1 TB SSD, Quadro RTX A4000 8 GB, Zustand A, Startpreis 0 €, ohne Gebote, Ende laut Snippet 5. August 2026 (abgelaufen) — [Huutokaupat](https://huutokaupat.com/en/kohde/6452113/dell-precision-7560-intel-core-i7-11850h-tehokannettava-15-6-fhd-32-gt-1024-ssd)
- Huutokaupat: Dell Precision 7560, 64 GB RAM, 1 TB SSD, RTX A4000 8 GB, Zustand B, Ende laut Snippet 31. Juli 2026 (abgelaufen) — [Huutokaupat](https://huutokaupat.com/en/kohde/6440770/dell-precision-7560-intel-core-i7-11850h-tehokannettava-15-6-fhd-64-gt-1024-ssd)
- Huutokaupat: Dell Precision 7540, Xeon E-2286M, 32 GB, RTX 3000 6 GB, Zustand A, Ende laut Snippet 6. September 2026 (abgelaufen) — [Huutokaupat](https://huutokaupat.com/en/kohde/6526337/dell-precision-7540-intel-xeon-e-2286m-tehokannettava-15-6-fhd-32-gt-1024-gt-ssd)
- Huutokaupat: Dell Precision 7550, i7-10875H, 32 GB, RTX 3000 6 GB, Zustand A, Ende laut Snippet 12. April 2026 (abgelaufen, veraltet) — [Huutokaupat](https://huutokaupat.com/en/kohde/6206378/dell-precision-7550-intel-core-i7-10875h-tehokannettava-15-6-fhd-32-gt-1024-ssd)
- Huutokaupat: Dell Precision 7550 i9-10885H, 32 GB, RTX 3000 6 GB, Zustand C — [Huutokaupat](https://huutokaupat.com/en/kohde/6471520/dell-precision-7550-intel-core-i9-10885h-tehokannettava-15-6-fhd-32-gt-1024-ssd)
- Intergavel (Ungarn), IT-Auktion: Lot 3 Dell Precision 7820 Workstation, Startpreis 3.000 €, zwei Xeon Gold 6252; Lot 9 Precision 7820, Startpreis 600 € — GPU-Ausstattung nicht angegeben — [Intergavel](https://intergavel.com/en/auctions/it-equipment-auction-2/3), [Intergavel](https://intergavel.com/en/auctions/it-equipment-auction-2/9)
- Alle Huutokaupat-Treffer sind Notebooks (15,6"), keine Desktop-GPUs oder Server. Für AI-Inferenz mit 8 GB VRAM (A4000) oder 6 GB (RTX 3000) praktisch unbrauchbar.
- eBay.de-Suchtreffer (Websuche, Snippets, nicht gelesen): Zu RTX 3090 vorwiegend Festpreisangebote und ein Auktionshinweis ohne klare Trennung — [eBay.de 187746102862](https://www.ebay.de/itm/187746102862), [eBay.de 17041884511](https://www.ebay.de/p/17041884511)
- Catawiki: Websuche fand keine RTX-4090-Lose — [Websuche-Ergebnis](https://extreme.pcgameshardware.de/threads/627828) (nur Charity- und Prototyp-Versteigerungen, nicht Catawiki).
- Einzige belegte Großauktion für RTX PRO 6000: RR Auction, signierte Karte (Jensen Huang) für 6.000 $ Schätzpreis, deutlich höher erzielt; Herkunft Verlosung GTC 2026 — [Drouot-Link](https://drouot.com/en/l/34469666-jensen-huang-twice-signed-nvidia-rtx-pro-6000-blackwell). Sammlerstück, nicht relevant für Inferenz-Beschaffung.
- Lot-Tiger: keine Treffer. US-Liquidator Tiger Group zeigt Workstation-Auktionen nur aus Jahren zurück — [Tiger Group](https://tigergroup.com/?p=12605).

### Inferences
- Für Inferenz-taugliche Karten (≥24 GB VRAM) sind die gefundenen Auktionsquellen praktisch leer. Die Relevanz liegt bei eBay-Auktionen (Privatverkäufer) und ggf. Restposten-Händlern, nicht bei den geprüften Auktionshäusern.
- Dell-Precision-Notebooks aus Auktionen sind für lokale Modelle (ab 7B quantisiert) zu VRAM-schwach.

### Gaps
- Keine direkte Prüfung von eBay.de, Catawiki, Troostwijk, Hood.de, auction.de, Maxauktion, Mister Auction oder Lot-Tiger möglich (403 bzw. DNS-Fehler). Aktuelle Oktober-2026-Lose sind daher **nicht verifiziert**.
- Keine Verkaufsdaten abgeschlossener eBay-Auktionen zugänglich.

## Frage 2: Welche aktuellen Lose sind interessant oder überteuert?

### Takeaway
Kein aktuelles Auktionslos konnte geprüft werden; eine Bewertung nach €/GB-VRAM ist für Oktober 2026 nicht möglich. Die einzigen auffindbaren Angebote für Desktop-GPUs stammen aus eBay.de-Snippets ohne Datum.

### Cited Findings
- RTX 3090 (24 GB), eBay.de-Snippets, undatiert: ASUS ROG Strix GeForce RTX 3090 OC 24 GB, 349 € + 14,69 € Versand; ASUS TUF Gaming RTX 3090, gebraucht, 1.039 € mit Käuferschutz, kostenloser Versand; GIGABYTE Gaming OC, Neuware, 2.700 € + 30 € Versand — [eBay.de](https://www.ebay.de/itm/187746102862), [eBay.de](https://www.ebay.de/p/17041884511). Das 349-€-Angebot ist ein Ausreißer, Zustand und Datum unbekannt.
- ASUS TUF RTX 3090 "nur Kühler" 75 € — nur Ersatzteil, für Inferenz ungeeignet — [Testsieger/eBay-Snippet](https://www.testsieger.de/next-gen-grafikkarten/7A5U11XC6BABG-nvidia-geforce-rtx-3090-founders-edition-24gb-gddr6-hdmi-3x-dp.html)
- RTX 4090 (24 GB), SweClockers-Privatanzeige, gebrauchte AORUS XTREME, 8.500 SEK nach Reparatur — Festpreis-Privatanzeige, kein Auktionslos, Datum unbekannt — [SweClockers](https://www.sweclockers.com/marknad/263654)
- A100 80 GB PCIe, eBay.de (Versand aus Japan), Einzelkarte ca. 15.700 € + ca. 255 € Versand, kein Rückgaberecht, Einfuhrzoll möglich, Datum unbekannt — [eBay.de 406086417179](https://www.ebay.de/itm/406086417179)
- Dual-A100-80GB-Server, eBay.de (UK), umgerechnet ca. 26.200 € — [eBay.de 277051254341](https://www.ebay.de/itm/277051254341)
- A100 80 GB refurbished "Grade A", Preis im Snippet nicht angegeben — [serverschmiede.com](https://www.serverschmiede.com/en/nvidia-tesla-ampere-a100-80gb-tensor-server-vdi-ki-gpu-pcie-x16-300w-1935-gbps)

### Inferences
- Der eBay-Einzelpreis der A100 80 GB (≈15.700 € + Versand, Japan) liegt in der Größenordnung des Neupreis-Leitfadens (PCIe 9.500–14.000 $, Stand 2025) und ist ohne Gewährleistung und mit Einfuhrrisiko kritisch zu sehen — [JarvisLabs](https://jarvislabs.ai/ai-faqs/nvidia-a100-gpu-price) (Neupreis-Orientierung, Stand 2025).
- Die 3090-Angebote 349 € vs. 1.039 € zeigen eine sehr große Spanne; das 349-€-Los sollte nur bei nachweislich funktionsfähigem Zustand als Schnäppchen gelten.

### Gaps
- Keine Lose mit Enddatum im Oktober 2026 verifiziert.
- Keine RTX 6000 Ada, RTX A6000, L40S oder H100 Auktionslose gefunden.

## Frage 3: Referenzpreise Gebrauchtmarkt

### Takeaway
Die belastbarste Referenz ist eine sekundäre Auswertung (tech-insider.org, Stand August 2026): Gebrauchte RTX 4090 median 2.104 €, RTX 5080 ca. 900–1.000 €. Für RTX PRO 6000 liegt eine Preisvergleichsseite mit Stand 3. Oktober 2026 bei 15.490 € (Tief) bzw. 12.192 € (12-Monats-Median). Für RTX 6000 Ada und L40S/H100 fehlen aktuelle EU-Gebrauchtpreise.

### Cited Findings
- RTX 4090 gebraucht: Median 2.104 € auf Kleinanzeigen und eBay, gemessen über PCPrice.Watch, August 2026; mittlere 50 % zwischen 2.031 und 2.189 €; UVP 2022 1.949 € — [tech-insider.org](https://tech-insider.org/de/rtx-4090-vs-rtx-5080-kaufen-oder-warten-2026/), [tech-insider.org Gebrauchtmarkt](https://tech-insider.org/rtx-4090-vs-rtx-5080-gebrauchtmarkt-2026/). Sekundärquelle, nicht unabhängig geprüft.
- RTX 5080 gebraucht 900–1.000 €, neu ca. 1.080–1.120 € — [tech-insider.org](https://tech-insider.org/rtx-4090-vs-rtx-5080-gebrauchtmarkt-2026/)
- RTX PRO 6000 Blackwell (96 GB): Preisvergleich Deutschland, Stand 3. Okt. 2026, niedrigster Preis 15.490 €, 12-Monats-Median 12.192 €, aktueller Preis ca. 11 % über 3-Monats-Median — [Hardwaredealz](https://www.hardwaredealz.com/hardware/gpu/nvidia-rtx-pro-6000-blackwell-workstation-edition-96gb/nvidia-96gb-gddr7-4x-dp-retail/preisvergleich.27857). Snippet, Seite nicht vollständig gelesen.
- RTX PRO 6000 Blackwell, Europa, Stand 19. Sept. 2026, Tief 15.390 € — [GPUPrix](https://gpuprix.com/eu/gpus/rtx-pro-6000-blackwell)
- RTX PRO 6000 Blackwell, Händler JACOB, ca. 11.300–11.600 € inkl. MwSt., geringe Stückzahl, Datum unklar — [JACOB](https://www.jacob.de/produkte/nvidia-quadro-rtx-900-5g144-2500-000-artnr-100670590.html)
- RTX PRO 6000 Blackwell, Nvidia-Marktplatz (Juni 2026), ca. 13.250 €, zu diesem Zeitpunkt nicht verfügbar — [Born City](https://borncity.com/news/rtx-pro-6000-blackwell-nvidias-workstation-kostet-jetzt-13-250-euro/)
- RTX PRO 6000 Neupreis (USA, März 2025): UVP 8.565 $ — [ITHome](https://www.ithome.com/0/963/977.htm)
- RTX 6000 Ada (48 GB): UVP 9.999 $ bei US-Listung (Tom's Hardware); ein australischer Händler listet ca. 12.307 AUD — [Tom's Hardware](https://www.tomshardware.com/news/rtx-6000-now-on-sale-8200), [area9](https://store.area9.com.au/products/nvidia-rtx-6000-ada-48gb-1898335). Kein EU-Gebrauchtpreis und kein Datum für den EU-Preis.
- A100 80 GB: siehe oben (eBay Japan ca. 15.700 € Einzelkarte, Datum unbekannt; Neupreis-Orientierung 9.500–14.000 $ PCIe, Stand 2025).

### Inferences
- Bei RTX PRO 6000 (96 GB) liegt der €/GB-Preis bei ca. 130–160 €/GB (15.490 € / 96 GB ≈ 161 €/GB; 11.300 € / 96 GB ≈ 118 €/GB). Bei RTX 4090 (24 GB) liegt er bei ca. 88 €/GB (2.104 € / 24 GB). Die RTX 4090 ist nach €/GB deutlich günstiger, aber mit 24 GB VRAM begrenzt.
- Ein eBay-Auktionspreis über dem Median von 2.104 € für eine 4090 ist ohne Zustandsnachweis überteuert.

### Gaps
- Keine idealo- oder Geizhals-Seiten für Gebrauchtware direkt abrufbar; die Idealo-Gebrauchtpreise für 4090 wurden nicht gefunden.
- Keine aktuellen EU-Gebrauchtpreise für RTX 6000 Ada, RTX A6000, L40S oder H100.
- eBay-Verkaufshistorie (abgeschlossene Auktionen) nicht zugänglich.

## Empfehlungen für weitere Recherche (außerhalb dieser Sitzung)
- Manuell die eBay.de-Suche "RTX 3090 / 4090" mit Filter "Auktion" und "Verkauft" prüfen; Enddatum ab 2026-10-01 notieren.
- Catawiki (Suche "GPU", "Grafikkarte", "Workstation") und Troostwijk (IT-Lots, Standort DE) direkt im Browser prüfen.
- Lot-Tiger, Hood.de und auction.de direkt prüfen; diese Plattformen waren hier nicht auffindbar.
- Referenzpreise gegen idealo/Geizhals-Gebrauchtlisten gegenprüfen (nicht erreichbar).
