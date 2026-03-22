# Task Type Classification

212 unique prompts across 30 task types (1-30).

| Task | Count | Description |
|------|-------|-------------|
| 1 | 7 | Opprett ansatt (Create employee) |
| 2 | 5 | Opprett kunde med org.nr (Create customer with org number) |
| 3 | 5 | Opprett produkt med MVA (Create product with VAT) |
| 4 | 8 | Opprett leverandør (Register supplier) |
| 5 | 8 | Opprett avdelinger (Create departments) |
| 6 | 5 | Opprett enkel faktura (Create and send simple invoice) |
| 7 | 5 | Registrer innkommende betaling (Register payment on customer invoice) |
| 8 | 5 | Opprett prosjekt knyttet til kunde (Create project linked to customer) |
| 9 | 7 | Faktura med flere produktlinjer og ulik MVA (Multi-line invoice with different VAT rates) |
| 10 | 9 | Ordre til faktura til betaling (Order to invoice to payment) |
| 11 | 8 | Registrer leverandørfaktura (Register supplier invoice) |
| 12 | 9 | Lønnskjøring med tillegg (Payroll with bonus) |
| 13 | 7 | Registrer reiseregning med utlegg (Travel expense with per diem) |
| 14 | 7 | Krediter faktura / kreditnota (Credit note) |
| 15 | 8 | Sett fastpris og fakturer prosjekt (Fixed price project + milestone invoice) |
| 16 | 7 | Timeføring og fakturering (Time logging + project invoice) |
| 17 | 6 | Opprett fri dimensjon og bokfør bilag (Custom dimension + post voucher) |
| 18 | 7 | Reverser betaling / bankretur (Reverse payment) |
| 19 | 5 | Ansatt fra arbeidskontrakt (PDF) (Employee from employment contract PDF) |
| 20 | 4 | Leverandørfaktura fra PDF (Supplier invoice from PDF) |
| 21 | 6 | Komplett onboarding fra tilbudsbrev (PDF) (Onboarding from offer letter PDF) |
| 22 | 10 | Utgift fra kvittering (PDF) på avdeling (Expense from receipt PDF to department) |
| 23 | 6 | Bankavstemming fra CSV (Bank reconciliation from CSV) |
| 24 | 9 | Feilsøking i hovedbok (General ledger error correction) |
| 25 | 10 | Forfallen faktura med purregebyr og delbetaling (Overdue invoice + reminder fee + partial payment) |
| 26 | 7 | Periodeavslutning / månedsavslutning (Monthly closing) |
| 27 | 10 | Valutafaktura med agio/disagio (Currency invoice with exchange rate difference) |
| 28 | 5 | Resultatanalyse, kostnadsøkning (Cost increase analysis + create internal projects) |
| 29 | 9 | Komplett prosjektsyklus med avvik (Complete project lifecycle) |
| 30 | 8 | Årsoppgjør (Simplified year-end closing) |
| **Total** | **212** | |

## Task 1: Opprett ansatt (Create employee)

**Count:** 7

**Fields (CreateEmployee):** `first_name`, `last_name`, `birth_year`, `birth_month`, `birth_day`, `email`, `start_year`, `start_month`, `start_day`

- Me har ein ny tilsett som heiter Torbjørn Neset, fødd 14. November 1991. Opprett vedkomande som tilsett med e-post torbjrn.neset@example.org og startdato 11. February 2026.
- Vi har en ny ansatt som heter Karin Berg, født 11. March 1992. Opprett vedkommende som ansatt med e-post karin.berg@example.org og startdato 29. March 2026.
- Me har ein ny tilsett som heiter Geir Stølsvik, fødd 6. March 1990. Opprett vedkomande som tilsett med e-post geir.stlsvik@example.org og startdato 14. November 2026.
- Nous avons un nouvel employé nommé Manon Thomas, né le 14. December 1981. Veuillez le créer en tant qu'employé avec l'e-mail manon.thomas@example.org et la date de début 22. January 2026.
- Tenemos un nuevo empleado llamado Lucía Rodríguez, nacido el 30. April 1996. Créelo como empleado con el correo lucia.rodriguez@example.org y fecha de inicio 22. April 2026.
- Vi har en ny ansatt som heter Lars Berg, født 10. April 1995. Opprett vedkommende som ansatt med e-post lars.berg@example.org og startdato 16. November 2026.
- Me har ein ny tilsett som heiter Gunnhild Eide, fødd 21. June 1997. Opprett vedkomande som tilsett med e-post gunnhild.eide@example.org og startdato 28. June 2026.

## Task 2: Opprett kunde med org.nr (Create customer with org number)

**Count:** 5

**Fields (CreateCustomer):** `customer_name`, `org_number`, `street_address`, `postal_code`, `city`, `email`

- Opprett kunden Snøhetta AS med organisasjonsnummer 969719878. Adressen er Industriveien 148, 2317 Hamar. E-post: post@snhetta.no.
- Crea el cliente Río Verde SL con número de organización 919234830. La dirección es Solveien 5, 4006 Stavanger. Correo: post@rio.no.
- Opprett kunden Nordhav AS med organisasjonsnummer 980461912. Adressen er Solveien 7, 4006 Stavanger. E-post: post@nordhav.no.
- Créez le client Colline SARL avec le numéro d'organisation 939137599. L'adresse est Kirkegata 77, 4611 Kristiansand. E-mail : post@colline.no.
- Crie o cliente Porto Alegre Lda com número de organização 834147254. O endereço é Storgata 65, 4611 Kristiansand. E-mail: post@porto.no.

## Task 3: Opprett produkt med MVA (Create product with VAT)

**Count:** 5

**Fields (CreateProduct):** `product_name`, `product_number`, `price_excl_vat`, `vat_rate`

- Opprett produktet "Frokostblanding" med produktnummer 1391. Prisen er 37450 kr eksklusiv MVA, og MVA-sats for næringsmidler på 15 % skal brukes.
- Crie o produto "Serviço de rede" com número de produto 4252. O preço é 8200 NOK sem IVA, utilizando a taxa padrão de 25 %.
- Opprett produktet "Analyserapport" med produktnummer 3637. Prisen er 31900 kr eksklusiv MVA, og standard MVA-sats på 25 % skal nyttast.
- Opprett produktet "Datarådgjeving" med produktnummer 4993. Prisen er 16250 kr eksklusiv MVA, og standard MVA-sats på 25 % skal nyttast.
- Create the product "Training Session" with product number 7908. The price is 26250 NOK excluding VAT, using the standard 25% VAT rate.

## Task 4: Opprett leverandør (Register supplier)

**Count:** 8

**Fields (CreateSupplier):** `supplier_name`, `org_number`, `email`

- Registrieren Sie den Lieferanten Brückentor GmbH mit der Organisationsnummer 867835989. E-Mail: faktura@brckentorgmbh.no.
- Registrieren Sie den Lieferanten Waldstein GmbH mit der Organisationsnummer 891505019. E-Mail: faktura@waldsteingmbh.no.
- Registe o fornecedor Floresta Lda com número de organização 883568885. E-mail: faktura@florestalda.no.
- Registrer leverandøren Fossekraft AS med organisasjonsnummer 977371635. E-post: faktura@fossekraft.no.
- Registe o fornecedor Luz do Sol Lda com número de organização 894596554. E-mail: faktura@luzdosollda.no.
- Registrer leverandøren Sjøbris AS med organisasjonsnummer 811212717. E-post: faktura@sjbris.no.
- Registe o fornecedor Oceano Lda com número de organização 841149394. E-mail: faktura@oceanolda.no.
- Register the supplier Silveroak Ltd with organization number 889586605. Email: faktura@silveroakltd.no.

## Task 5: Opprett avdelinger (Create departments)

**Count:** 8

**Fields (CreateDepartments):** `department_names`

- Crie três departamentos no Tripletex: "Drift", "Kundeservice" e "HR".
- Opprett tre avdelinger i Tripletex: "Utvikling", "Drift" og "HR".
- Créez trois départements dans Tripletex : "Produksjon", "Salg" et "Logistikk".
- Créez trois départements dans Tripletex : "Administrasjon", "IT" et "Utvikling".
- Créez trois départements dans Tripletex : "Produksjon", "Salg" et "Innkjøp".
- Crea tres departamentos en Tripletex: "Logistikk", "Produksjon" y "Drift".
- Créez trois départements dans Tripletex : "Økonomi", "Lager" et "IT".
- Crea tres departamentos en Tripletex: "IT", "Drift" y "Kundeservice".

## Task 6: Opprett enkel faktura (Create and send simple invoice)

**Count:** 5

**Fields (CreateAndSendInvoice):** `customer_name`, `org_number`, `amount_excl_vat`, `product_description`

- Crea y envía una factura al cliente Luna SL (org. nº 844920520) por 20200 NOK sin IVA. La factura es por Servicio de red.
- Opprett og send en faktura til kunden Polaris AS (org.nr 963373937) på 32600 kr eksklusiv MVA. Fakturaen gjelder Webdesign.
- Create and send an invoice to the customer Clearwater Ltd (org no. 935400759) for 3100 NOK excluding VAT. The invoice is for Data Advisory.
- Crie e envie uma fatura ao cliente Porto Alegre Lda (org. nº 826870192) por 22700 NOK sem IVA. A fatura refere-se a Design web.
- Crea y envía una factura al cliente Montaña SL (org. nº 831306742) por 48600 NOK sin IVA. La factura es por Licencia de software.

## Task 7: Registrer innkommende betaling (Register payment on customer invoice)

**Count:** 5

**Fields (RegisterPayment):** `customer_name`, `org_number`, `amount_excl_vat`, `invoice_description`

- El cliente Viento SL (org. nº 908616537) tiene una factura pendiente de 37850 NOK sin IVA por "Mantenimiento". Registre el pago completo de esta factura.
- The customer Ironbridge Ltd (org no. 985423849) has an outstanding invoice for 32900 NOK excluding VAT for "Web Design". Register full payment on this invoice.
- Der Kunde Sonnental GmbH (Org.-Nr. 855482207) hat eine offene Rechnung über 33500 NOK ohne MwSt. für "Wartung". Registrieren Sie die vollständige Zahlung dieser Rechnung.
- Kunden Strandvik AS (org.nr 840390055) har ein uteståande faktura på 27050 kr eksklusiv MVA for "Datarådgjeving". Registrer full betaling på denne fakturaen.
- El cliente Costa Brava SL (org. nº 833355937) tiene una factura pendiente de 45700 NOK sin IVA por "Asesoría de datos". Registre el pago completo de esta factura.

## Task 8: Opprett prosjekt knyttet til kunde (Create project linked to customer)

**Count:** 5

**Fields (CreateProject):** `project_name`, `customer_name`, `org_number`, `project_manager_first_name`, `project_manager_last_name`, `project_manager_email`

- Crie o projeto "Análise Porto" vinculado ao cliente Porto Alegre Lda (org. nº 996943305). O gerente de projeto é Lucas Oliveira (lucas.oliveira@example.org).
- Opprett prosjektet "Migrasjon Snøhetta" knyttet til kunden Snøhetta AS (org.nr 842796032). Prosjektleder er Eline Berg (eline.berg@example.org).
- Opprett prosjektet "Analyse Fjordkraft" knyttet til kunden Fjordkraft AS (org.nr 944845712). Prosjektleder er Hilde Johansen (hilde.johansen@example.org).
- Opprett prosjektet "Integrasjon Vestfjord" knytt til kunden Vestfjord AS (org.nr 954285405). Prosjektleiar er Håkon Eide (hakon.eide@example.org).
- Erstellen Sie das Projekt "Integration Bergwerk" verknüpft mit dem Kunden Bergwerk GmbH (Org.-Nr. 986555080). Projektleiter ist Leon Meyer (leon.meyer@example.org).

## Task 9: Faktura med flere produktlinjer og ulik MVA (Multi-line invoice with different VAT rates)

**Count:** 7

**Fields (MultiLineInvoice):** `customer_name`, `org_number`, `product_lines`

**Fields (ProductLine):** `product_name`, `product_number`, `amount_excl_vat`, `vat_rate`

- Opprett ein faktura til kunden Strandvik AS (org.nr 900314183) med tre produktlinjer: Webdesign (9716) til 13450 kr med 25 % MVA, Skylagring (6906) til 7700 kr med 15 % MVA (næringsmiddel), og Systemutvikling (2265) til 18800 kr med 0 % MVA (avgiftsfri).
- Crie uma fatura para o cliente Montanha Lda (org. nº 869972401) com três linhas de produto: Sessão de formação (7733) a 22950 NOK com 25 % IVA, Licença de software (6106) a 10250 NOK com 15 % IVA (alimentos), e Manutenção (1351) a 3150 NOK com 0 % IVA (isento).
- Créez une facture pour le client Océan SARL (nº org. 974909103) avec trois lignes de produit : Développement système (9068) à 11000 NOK avec 25 % TVA, Licence logicielle (3111) à 7350 NOK avec 15 % TVA (alimentaire), et Session de formation (9564) à 13150 NOK avec 0 % TVA (exonéré).
- Erstellen Sie eine Rechnung für den Kunden Nordlicht GmbH (Org.-Nr. 855854171) mit drei Produktzeilen: Netzwerkdienst (2450) zu 28650 NOK mit 25 % MwSt., Cloud-Speicher (6871) zu 13750 NOK mit 15 % MwSt. (Lebensmittel), und Wartung (2881) zu 18000 NOK mit 0 % MwSt. (befreit).
- Crea una factura para el cliente Sierra SL (org. nº 861379760) con tres líneas de producto: Mantenimiento (2109) a 27500 NOK con 25 % IVA, Horas de consultoría (1175) a 3900 NOK con 15 % IVA (alimentos), y Informe de análisis (9974) a 3400 NOK con 0 % IVA (exento).
- Crea una factura para el cliente Río Verde SL (org. nº 863477905) con tres líneas de producto: Desarrollo de sistemas (2376) a 12000 NOK con 25 % IVA, Asesoría de datos (1496) a 13450 NOK con 15 % IVA (alimentos), y Mantenimiento (4543) a 12050 NOK con 0 % IVA (exento).
- Crie uma fatura para o cliente Solmar Lda (org. nº 857302435) com três linhas de produto: Design web (4982) a 21250 NOK com 25 % IVA, Relatório de análise (8365) a 7100 NOK com 15 % IVA (alimentos), e Sessão de formação (1064) a 9550 NOK com 0 % IVA (isento).

## Task 10: Ordre til faktura til betaling (Order to invoice to payment)

**Count:** 9

**Fields (OrderProduct):** `product_name`, `product_number`, `price`

**Fields (OrderToInvoice):** `customer_name`, `org_number`, `products`

- Crie um pedido para o cliente Estrela Lda (org. nº 842487803) com os produtos Design web (1851) a 24050 NOK e Consultoria de dados (5065) a 13450 NOK. Converta o pedido em fatura e registe o pagamento total.
- Opprett en ordre for kunden Fjordkraft AS (org.nr 911511053) med produktene Opplæring (7579) til 14650 kr og Webdesign (2292) til 11800 kr. Konverter ordren til faktura og registrer full betaling.
- Crie um pedido para o cliente Horizonte Lda (org. nº 904130338) com os produtos Serviço de rede (6247) a 15250 NOK e Desenvolvimento de sistemas (5919) a 13250 NOK. Converta o pedido em fatura e registe o pagamento total.
- Crea un pedido para el cliente Río Verde SL (org. nº 937237243) con los productos Informe de análisis (5700) a 33200 NOK y Diseño web (2680) a 17200 NOK. Convierte el pedido en factura y registra el pago completo.
- Crea un pedido para el cliente Dorada SL (org. nº 984411359) con los productos Desarrollo de sistemas (5240) a 21950 NOK y Sesión de formación (5871) a 6350 NOK. Convierte el pedido en factura y registra el pago completo.
- Opprett ein ordre for kunden Bølgekraft AS (org.nr 908252764) med produkta Nettverksteneste (6065) til 25500 kr og Systemutvikling (2511) til 23550 kr. Konverter ordren til faktura og registrer full betaling.
- Erstellen Sie einen Auftrag für den Kunden Waldstein GmbH (Org.-Nr. 899060113) mit den Produkten Netzwerkdienst (5411) zu 29200 NOK und Schulung (7883) zu 10350 NOK. Wandeln Sie den Auftrag in eine Rechnung um und registrieren Sie die vollständige Zahlung.
- Crea un pedido para el cliente Río Verde SL (org. nº 951612936) con los productos Informe de análisis (4430) a 20900 NOK y Sesión de formación (7773) a 18350 NOK. Convierte el pedido en factura y registra el pago completo.
- Opprett en ordre for kunden Stormberg AS (org.nr 870531559) med produktene Vedlikehold (4665) til 35200 kr og Systemutvikling (7431) til 4400 kr. Konverter ordren til faktura og registrer full betaling.

## Task 11: Registrer leverandørfaktura (Register supplier invoice)

**Count:** 8

**Fields (RegisterSupplierInvoice):** `invoice_number`, `supplier_name`, `org_number`, `amount_incl_vat`, `expense_account`, `vat_rate`

- We have received invoice INV-2026-9075 from the supplier Brightstone Ltd (org no. 890932991) for 59800 NOK including VAT. The amount relates to office services (account 6300). Register the supplier invoice with the correct input VAT (25%).
- We have received invoice INV-2026-3749 from the supplier Ridgepoint Ltd (org no. 902484981) for 65850 NOK including VAT. The amount relates to office services (account 6590). Register the supplier invoice with the correct input VAT (25%).
- Recebemos a fatura INV-2026-6556 do fornecedor Solmar Lda (org. nº 974178680) no valor de 50750 NOK com IVA incluído. O montante refere-se a serviços de escritório (conta 6500). Registe a fatura do fornecedor com o IVA dedutível correto (25 %).
- Nous avons reçu la facture INV-2026-4647 du fournisseur Cascade SARL (nº org. 951586935) de 30000 NOK TTC. Le montant concerne des services de bureau (compte 6860). Enregistrez la facture fournisseur avec la TVA déductible correcte (25 %).
- We have received invoice INV-2026-8735 from the supplier Brightstone Ltd (org no. 913701585) for 8500 NOK including VAT. The amount relates to office services (account 7100). Register the supplier invoice with the correct input VAT (25%).
- We have received invoice INV-2026-3205 from the supplier Ironbridge Ltd (org no. 828254375) for 24500 NOK including VAT. The amount relates to office services (account 6590). Register the supplier invoice with the correct input VAT (25%).
- Recebemos a fatura INV-2026-6293 do fornecedor Montanha Lda (org. nº 980979431) no valor de 12050 NOK com IVA incluído. O montante refere-se a serviços de escritório (conta 7000). Registe a fatura do fornecedor com o IVA dedutível correto (25 %).
- Recebemos a fatura INV-2026-4855 do fornecedor Solmar Lda (org. nº 972752843) no valor de 62600 NOK com IVA incluído. O montante refere-se a serviços de escritório (conta 6860). Registe a fatura do fornecedor com o IVA dedutível correto (25 %).

## Task 12: Lønnskjøring med tillegg (Payroll with bonus)

**Count:** 9

**Fields (PayrollWithBonus):** `first_name`, `last_name`, `email`, `base_salary`, `bonus_amount`

- Run payroll for William Taylor (william.taylor@example.org) for this month. The base salary is 39400 NOK. Add a one-time bonus of 11800 NOK on top of the base salary. If the salary API is unavailable, you can use manual vouchers on salary accounts (5000-series) to record the payroll expense.
- Køyr løn for Brita Stølsvik (brita.stlsvik@example.org) for denne månaden. Grunnløn er 36000 kr. Legg til ein eingongsbonus på 15400 kr i tillegg til grunnløna.
- Køyr løn for Eirik Lunde (eirik.lunde@example.org) for denne månaden. Grunnløn er 39100 kr. Legg til ein eingongsbonus på 8200 kr i tillegg til grunnløna.
- Køyr løn for Eirik Brekke (eirik.brekke@example.org) for denne månaden. Grunnløn er 41050 kr. Legg til ein eingongsbonus på 9800 kr i tillegg til grunnløna.
- Processe o salário de Rafael Silva (rafael.silva@example.org) para este mês. O salário base é de 41150 NOK. Adicione um bónus único de 8500 NOK além do salário base.
- Kjør lønn for Tor Bakken (tor.bakken@example.org) for denne måneden. Grunnlønn er 34100 kr. Legg til en engangsbonus på 6550 kr i tillegg til grunnlønnen. Dersom lønns-API-et ikke fungerer, kan du bruke manuelle bilag på lønnskontoer (5000-serien) for å registrere lønnskostnaden.
- Führen Sie die Gehaltsabrechnung für Sophia Müller (sophia.muller@example.org) für diesen Monat durch. Das Grundgehalt beträgt 48350 NOK. Fügen Sie einen einmaligen Bonus von 15450 NOK zum Grundgehalt hinzu.
- Exécutez la paie de Sarah Moreau (sarah.moreau@example.org) pour ce mois. Le salaire de base est de 56900 NOK. Ajoutez une prime unique de 15800 NOK en plus du salaire de base.
- Ejecute la nómina de Fernando Sánchez (fernando.sanchez@example.org) para este mes. El salario base es de 42350 NOK. Añada una bonificación única de 12850 NOK además del salario base.

## Task 13: Registrer reiseregning med utlegg (Travel expense with per diem)

**Count:** 7

**Fields (TravelExpense):** `first_name`, `last_name`, `email`, `trip_description`, `trip_destination`, `days`, `daily_rate`, `flight_cost`, `taxi_cost`

- Registrer ei reiserekning for Marit Vik (marit.vik@example.org) for "Konferanse Kristiansand". Reisa varte 4 dagar med diett (dagssats 800 kr). Utlegg: flybillett 7600 kr og taxi 600 kr.
- Registrer ei reiserekning for Svein Berge (svein.berge@example.org) for "Kundebesøk Trondheim". Reisa varte 5 dagar med diett (dagssats 800 kr). Utlegg: flybillett 2850 kr og taxi 200 kr.
- Registrer ei reiserekning for Svein Vik (svein.vik@example.org) for "Kundebesøk Bergen". Reisa varte 5 dagar med diett (dagssats 800 kr). Utlegg: flybillett 7900 kr og taxi 250 kr.
- Registrer en reiseregning for Ingrid Larsen (ingrid.larsen@example.org) for "Kundebesøk Trondheim". Reisen varte 2 dager med diett (dagsats 800 kr). Utlegg: flybillett 2500 kr og taxi 600 kr.
- Registrer en reiseregning for Magnus Haugen (magnus.haugen@example.org) for "Kundebesøk Bergen". Reisen varte 4 dager med diett (dagsats 800 kr). Utlegg: flybillett 5050 kr og taxi 750 kr.
- Register a travel expense for Charlotte Williams (charlotte.williams@example.org) for "Client visit Bodø". The trip lasted 3 days with per diem (daily rate 800 NOK). Expenses: flight ticket 6200 NOK and taxi 400 NOK.
- Register a travel expense for William Wilson (william.wilson@example.org) for "Client visit Trondheim". The trip lasted 2 days with per diem (daily rate 800 NOK). Expenses: flight ticket 7600 NOK and taxi 700 NOK.

## Task 14: Krediter faktura / kreditnota (Credit note)

**Count:** 7

**Fields (CreditNote):** `customer_name`, `org_number`, `invoice_description`, `amount_excl_vat`

- Kunden Vestfjord AS (org.nr 860678403) har reklamert på fakturaen for "Skylagring" (45350 kr ekskl. MVA). Opprett ei fullstendig kreditnota som reverserer heile fakturaen.
- O cliente Cascata Lda (org. nº 967090743) reclamou sobre a fatura referente a "Sessão de formação" (23800 NOK sem IVA). Emita uma nota de crédito completa que reverta toda a fatura.
- El cliente Viento SL (org. nº 997137310) ha reclamado sobre la factura por "Desarrollo de sistemas" (47700 NOK sin IVA). Emita una nota de crédito completa que revierta toda la factura.
- Le client Montagne SARL (nº org. 882988155) a réclamé concernant la facture pour "Heures de conseil" (40900 NOK HT). Émettez un avoir complet qui annule l'intégralité de la facture.
- El cliente Luna SL (org. nº 993794775) ha reclamado sobre la factura por "Diseño web" (15150 NOK sin IVA). Emita una nota de crédito completa que revierta toda la factura.
- El cliente Viento SL (org. nº 857019199) ha reclamado sobre la factura por "Informe de análisis" (27200 NOK sin IVA). Emita una nota de crédito completa que revierta toda la factura.
- Kunden Nordlys AS (org.nr 902392165) har reklamert på fakturaen for "Programvarelisens" (47350 kr ekskl. MVA). Opprett ei fullstendig kreditnota som reverserer heile fakturaen.

## Task 15: Sett fastpris og fakturer prosjekt (Fixed price project + milestone invoice)

**Count:** 8

**Fields (FixedPriceProject):** `fixed_price`, `project_name`, `customer_name`, `org_number`, `project_manager_first_name`, `project_manager_last_name`, `project_manager_email`, `invoice_percentage`

- Fixez un prix forfaitaire de 128500 NOK sur le projet "Mise à niveau infrastructure" pour Forêt SARL (nº org. 846715363). Le chef de projet est Camille Martin (camille.martin@example.org). Facturez au client 33 % du prix fixe comme paiement d'étape.
- Set a fixed price of 170500 NOK on the project "Infrastructure Upgrade" for Brightstone Ltd (org no. 850116091). The project manager is Charlotte Walker (charlotte.walker@example.org). Invoice the customer for 33% of the fixed price as a milestone payment.
- Legen Sie einen Festpreis von 473250 NOK für das Projekt "Datensicherheit" für Windkraft GmbH (Org.-Nr. 886395582) fest. Projektleiter ist Maximilian Wagner (maximilian.wagner@example.org). Stellen Sie dem Kunden 25 % des Festpreises als Meilensteinzahlung in Rechnung.
- Fixez un prix forfaitaire de 328650 NOK sur le projet "Développement e-commerce" pour Cascade SARL (nº org. 913754689). Le chef de projet est Alice Moreau (alice.moreau@example.org). Facturez au client 25 % du prix fixe comme paiement d'étape.
- Sett fastpris 181650 kr på prosjektet "Nettbutikk-utvikling" for Tindra AS (org.nr 870827946). Prosjektleder er Kristian Nilsen (kristian.nilsen@example.org). Fakturer kunden for 50 % av fastprisen som en delbetaling.
- Legen Sie einen Festpreis von 350650 NOK für das Projekt "ERP-Implementierung" für Sonnental GmbH (Org.-Nr. 877407047) fest. Projektleiter ist Finn Müller (finn.muller@example.org). Stellen Sie dem Kunden 25 % des Festpreises als Meilensteinzahlung in Rechnung.
- Establezca un precio fijo de 266550 NOK en el proyecto "Seguridad de datos" para Montaña SL (org. nº 865036981). El director del proyecto es Sofía Pérez (sofia.perez@example.org). Facture al cliente el 50 % del precio fijo como pago parcial.
- Sett fastpris 318800 kr på prosjektet "Digital transformasjon" for Strandvik AS (org.nr 883822684). Prosjektleiar er Jorunn Brekke (jorunn.brekke@example.org). Fakturer kunden for 50 % av fastprisen som ei delbetaling.

## Task 16: Timeføring og fakturering (Time logging + project invoice)

**Count:** 7

**Fields (TimeLoggingInvoice):** `hours`, `first_name`, `last_name`, `email`, `activity_name`, `project_name`, `customer_name`, `org_number`, `hourly_rate`

- Erfassen Sie 18 Stunden für Sophia Schmidt (sophia.schmidt@example.org) auf der Aktivität "Design" im Projekt "Sicherheitsaudit" für Windkraft GmbH (Org.-Nr. 882984826). Stundensatz: 950 NOK/h. Erstellen Sie eine Projektrechnung an den Kunden basierend auf den erfassten Stunden.
- Registe 4 horas para Maria Ferreira (maria.ferreira@example.org) na atividade "Utvikling" do projeto "Desenvolvimento de app" para Estrela Lda (org. nº 909621682). Taxa horária: 1050 NOK/h. Gere uma fatura de projeto ao cliente com base nas horas registadas.
- Registe 35 horas para Inês Ferreira (ines.ferreira@example.org) na atividade "Testing" do projeto "Configuração cloud" para Floresta Lda (org. nº 949247589). Taxa horária: 1000 NOK/h. Gere uma fatura de projeto ao cliente com base nas horas registadas.
- Log 15 hours for Samuel Williams (samuel.williams@example.org) on the activity "Analyse" in the project "Website Redesign" for Windmill Ltd (org no. 898523942). Hourly rate: 1950 NOK/h. Generate a project invoice to the customer based on the logged hours.
- Registe 27 horas para Ana Sousa (ana.sousa@example.org) na atividade "Utvikling" do projeto "Migração de dados" para Montanha Lda (org. nº 870501366). Taxa horária: 800 NOK/h. Gere uma fatura de projeto ao cliente com base nas horas registadas.
- Log 34 hours for Charlotte Williams (charlotte.williams@example.org) on the activity "Analyse" in the project "Security Audit" for Windmill Ltd (org no. 851492623). Hourly rate: 1300 NOK/h. Generate a project invoice to the customer based on the logged hours.
- Erfassen Sie 18 Stunden für Mia Meyer (mia.meyer@example.org) auf der Aktivität "Testing" im Projekt "Sicherheitsaudit" für Nordlicht GmbH (Org.-Nr. 934651995). Stundensatz: 1400 NOK/h. Erstellen Sie eine Projektrechnung an den Kunden basierend auf den erfassten Stunden.

## Task 17: Opprett fri dimensjon og bokfør bilag (Custom dimension + post voucher)

**Count:** 6

**Fields (CustomDimension):** `dimension_name`, `dimension_values`, `account_number`, `amount`, `linked_dimension_value`

- Cree una dimensión contable personalizada "Produktlinje" con los valores "Basis" y "Premium". Luego registre un asiento en la cuenta 7140 por 39600 NOK, vinculado al valor de dimensión "Premium".
- Opprett ein fri rekneskapsdimensjon "Marked" med verdiane "Offentlig" og "Privat". Bokfør deretter eit bilag på konto 7300 for 49300 kr, knytt til dimensjonsverdien "Offentlig".
- Opprett ein fri rekneskapsdimensjon "Prosjekttype" med verdiane "Utvikling" og "Internt". Bokfør deretter eit bilag på konto 6860 for 40600 kr, knytt til dimensjonsverdien "Utvikling".
- Erstellen Sie eine benutzerdefinierte Buchhaltungsdimension "Kostsenter" mit den Werten "Kundeservice" und "Markedsføring". Buchen Sie dann einen Beleg auf Konto 7100 über 42400 NOK, verknüpft mit dem Dimensionswert "Markedsføring".
- Erstellen Sie eine benutzerdefinierte Buchhaltungsdimension "Prosjekttype" mit den Werten "Internt" und "Utvikling". Buchen Sie dann einen Beleg auf Konto 6340 über 44500 NOK, verknüpft mit dem Dimensionswert "Internt".
- Opprett ein fri rekneskapsdimensjon "Produktlinje" med verdiane "Basis" og "Avansert". Bokfør deretter eit bilag på konto 6340 for 15000 kr, knytt til dimensjonsverdien "Avansert".

## Task 18: Reverser betaling / bankretur (Reverse payment)

**Count:** 7

**Fields (ReversePayment):** `customer_name`, `org_number`, `invoice_description`, `amount_excl_vat`

- Le paiement de Colline SARL (nº org. 916057903) pour la facture "Maintenance" (8300 NOK HT) a été retourné par la banque. Annulez le paiement afin que la facture affiche à nouveau le montant impayé.
- The payment from Ridgepoint Ltd (org no. 990845042) for the invoice "Cloud Storage" (43550 NOK excl. VAT) was returned by the bank. Reverse the payment so the invoice shows the outstanding amount again.
- Die Zahlung von Brückentor GmbH (Org.-Nr. 944848479) für die Rechnung "Wartung" (42200 NOK ohne MwSt.) wurde von der Bank zurückgebucht. Stornieren Sie die Zahlung, damit die Rechnung wieder den offenen Betrag anzeigt.
- Die Zahlung von Nordlicht GmbH (Org.-Nr. 985405077) für die Rechnung "Wartung" (42200 NOK ohne MwSt.) wurde von der Bank zurückgebucht. Stornieren Sie die Zahlung, damit die Rechnung wieder den offenen Betrag anzeigt.
- Le paiement de Étoile SARL (nº org. 943745862) pour la facture "Conseil en données" (33900 NOK HT) a été retourné par la banque. Annulez le paiement afin que la facture affiche à nouveau le montant impayé.
- Betalingen fra Polaris AS (org.nr 896496468) for fakturaen "Skylagring" (17200 kr ekskl. MVA) ble returnert av banken. Reverser betalingen slik at fakturaen igjen viser utestående beløp.
- El pago de Solmar SL (org. nº 836598741) por la factura "Licencia de software" (8000 NOK sin IVA) fue devuelto por el banco. Revierta el pago para que la factura vuelva a mostrar el importe pendiente.

## Task 19: Ansatt fra arbeidskontrakt (PDF) (Employee from employment contract PDF)

**Count:** 5

**Fields (EmployeeFromContractPDF):** `first_name`, `last_name`, `birth_year`, `birth_month`, `birth_day`, `national_identity_number`, `email`, `bank_account_number`, `department`, `occupation_code`, `employment_percentage`, `annual_salary`, `start_year`, `start_month`, `start_day`

- Vous avez recu un contrat de travail (voir PDF ci-joint). Creez l'employe dans Tripletex avec tous les details du contrat : numero d'identite nationale, date de naissance, departement, code de profession, salaire, pourcentage d'emploi et date de debut.
- Du har motteke ein arbeidskontrakt (sjaa vedlagt PDF). Opprett den tilsette i Tripletex med alle detaljar fraa kontrakten: personnummer, fodselsdato, avdeling, stillingskode, lonn, stillingsprosent og startdato.
- Voce recebeu um contrato de trabalho (ver PDF anexo). Crie o funcionario no Tripletex com todos os detalhes do contrato: numero de identidade nacional, data de nascimento, departamento, codigo de ocupacao, salario, percentagem de emprego e data de inicio.
- You received an employment contract (see attached PDF). Create the employee in Tripletex with all details from the contract: national identity number, date of birth, department, occupation code, salary, employment percentage, and start date.
- Sie haben einen Arbeitsvertrag erhalten (siehe beigefugte PDF). Erstellen Sie den Mitarbeiter in Tripletex mit allen Details aus dem Vertrag: Personalnummer, Geburtsdatum, Abteilung, Berufsschluessel, Gehalt, Beschaeftigungsprozentsatz und Startdatum.

## Task 20: Leverandørfaktura fra PDF (Supplier invoice from PDF)

**Count:** 4

**Fields (SupplierInvoiceFromPDF):** `supplier_name`, `org_number`, `supplier_street_address`, `supplier_postal_code`, `supplier_city`, `supplier_bank_account`, `invoice_number`, `invoice_year`, `invoice_month`, `invoice_day`, `due_year`, `due_month`, `due_day`, `description`, `amount_excl_vat`, `vat_rate`, `amount_incl_vat`, `expense_account`

- Voce recebeu uma fatura de fornecedor (ver PDF anexo). Registe a fatura no Tripletex. Crie o fornecedor se nao existir. Use a conta de despesas correta e o IVA de entrada.
- Du har mottatt en leverandorfaktura (se vedlagt PDF). Registrer fakturaen i Tripletex. Opprett leverandoren hvis den ikke finnes. Bruk riktig utgiftskonto og inngaende MVA.
- Du har motteke ein leverandorfaktura (sjaa vedlagt PDF). Registrer fakturaen i Tripletex. Opprett leverandoren viss den ikkje finst. Bruk rett utgiftskonto og inngaaande MVA.
- Vous avez recu une facture fournisseur (voir PDF ci-joint). Enregistrez la facture dans Tripletex. Creez le fournisseur s'il n'existe pas. Utilisez le bon compte de charges et la TVA deductible.

## Task 21: Komplett onboarding fra tilbudsbrev (PDF) (Onboarding from offer letter PDF)

**Count:** 6

**Fields (OnboardingFromOfferPDF):** `first_name`, `last_name`, `birth_year`, `birth_month`, `birth_day`, `job_title`, `department`, `employment_percentage`, `annual_salary`, `standard_work_hours_per_day`, `start_year`, `start_month`, `start_day`

- Vous avez recu une lettre d'offre (voir PDF ci-joint) pour un nouvel employe. Effectuez l'integration complete : creez l'employe, attribuez le bon departement, configurez les details d'emploi avec le pourcentage et le salaire annuel, et configurez les heures de travail standard.
- Sie haben ein Angebotsschreiben erhalten (siehe beigefugte PDF) fuer einen neuen Mitarbeiter. Fuehren Sie das vollstaendige Onboarding durch: erstellen Sie den Mitarbeiter, weisen Sie die richtige Abteilung zu, richten Sie die Beschaeftigungsdetails mit Prozentsatz und Jahresgehalt ein, und konfigurieren Sie die Standardarbeitszeit.
- Has recibido una carta de oferta (ver PDF adjunto) para un nuevo empleado. Completa la incorporacion: crea el empleado, asigna el departamento correcto, configura los detalles de empleo con porcentaje y salario anual, y configura las horas de trabajo estandar.
- Du har motteke eit tilbodsbrev (sjaa vedlagt PDF) for ein ny tilsett. Utfor komplett onboarding: opprett den tilsette, tilknytt rett avdeling, set opp tilsetjingsforhold med stillingsprosent og arslonn, og konfigurer standard arbeidstid.
- Du har mottatt et tilbudsbrev (se vedlagt PDF) for en ny ansatt. Utfor komplett onboarding: opprett den ansatte, tilknytt riktig avdeling, sett opp ansettelsesforhold med stillingsprosent og arslonn, og konfigurer standard arbeidstid.
- Voce recebeu uma carta de oferta (ver PDF anexo) para um novo funcionario. Complete a integracao: crie o funcionario, atribua o departamento correto, configure os detalhes de emprego com percentagem e salario anual, e configure as horas de trabalho padrao.

## Task 22: Utgift fra kvittering (PDF) på avdeling (Expense from receipt PDF to department)

**Count:** 10

**Fields (ExpenseFromReceiptPDF):** `expense_item_name`, `department_name`, `item_price`, `receipt_year`, `receipt_month`, `receipt_day`, `store_name`, `store_street_address`, `store_postal_code`, `store_city`, `store_org_number`, `total_amount`, `vat_amount`

- Vi trenger USB-hub fra denne kvitteringen bokfort pa avdeling HR. Bruk riktig utgiftskonto basert pa kjopet, og sorg for korrekt MVA-behandling.
- We need the Togbillett expense from this receipt posted to department Produksjon. Use the correct expense account and ensure proper VAT treatment.
- Vi treng Oppbevaringsboks fra denne kvitteringa bokfort pa avdeling Regnskap. Bruk rett utgiftskonto basert pa kjopet, og sorg for korrekt MVA-behandling.
- We need the Kontorstoler expense from this receipt posted to department Økonomi. Use the correct expense account and ensure proper VAT treatment.
- Precisamos da despesa de Kaffemøte deste recibo registada no departamento Salg. Use a conta de despesas correta e garanta o tratamento correto do IVA.
- Vi treng USB-hub fra denne kvitteringa bokfort pa avdeling Kvalitetskontroll. Bruk rett utgiftskonto basert pa kjopet, og sorg for korrekt MVA-behandling.
- Nous avons besoin de la depense Oppbevaringsboks de ce recu enregistree au departement Produksjon. Utilisez le bon compte de charges et assurez le traitement correct de la TVA.
- Wir benotigen die Headset-Ausgabe aus dieser Quittung in der Abteilung Salg. Verwenden Sie das richtige Aufwandskonto und stellen Sie die korrekte MwSt.-Behandlung sicher.
- Necesitamos el gasto de USB-hub de este recibo registrado en el departamento Utvikling. Usa la cuenta de gastos correcta y asegura el tratamiento correcto del IVA.
- Precisamos da despesa de Overnatting deste recibo registada no departamento Utvikling. Use a conta de despesas correta e garanta o tratamento correto do IVA.

## Task 23: Bankavstemming fra CSV (Bank reconciliation from CSV)

**Count:** 6

**Fields (BankFee):** `year`, `month`, `day`, `amount`

**Fields (BankReconciliation):** `incoming_payments`, `outgoing_payments`, `bank_fees`, `interest`

**Fields (IncomingPayment):** `year`, `month`, `day`, `customer_name`, `invoice_number`, `amount`

**Fields (Interest):** `year`, `month`, `day`, `amount`

**Fields (OutgoingPayment):** `year`, `month`, `day`, `supplier_name`, `amount`

- Reconcilie o extrato bancario (CSV anexo) com as faturas em aberto no Tripletex. Relacione os pagamentos recebidos com as faturas de clientes e os pagamentos efetuados com as faturas de fornecedores. Trate os pagamentos parciais corretamente.
- Avstem bankutskrifta (vedlagt CSV) mot opne fakturaer i Tripletex. Match innbetalingar til kundefakturaer og utbetalingar til leverandorfakturaer. Handter delbetalingar korrekt.
- Gleichen Sie den Kontoauszug (beigefuegte CSV) mit den offenen Rechnungen in Tripletex ab. Ordnen Sie eingehende Zahlungen Kundenrechnungen und ausgehende Zahlungen Lieferantenrechnungen zu. Behandeln Sie Teilzahlungen korrekt.
- Concilia el extracto bancario (CSV adjunto) con las facturas abiertas en Tripletex. Relaciona los pagos entrantes con las facturas de clientes y los pagos salientes con las facturas de proveedores. Maneja los pagos parciales correctamente.
- Rapprochez le releve bancaire (CSV ci-joint) avec les factures ouvertes dans Tripletex. Associez les paiements entrants aux factures clients et les paiements sortants aux factures fournisseurs. Gerez correctement les paiements partiels.
- Reconcile the bank statement (attached CSV) against open invoices in Tripletex. Match incoming payments to customer invoices and outgoing payments to supplier invoices. Handle partial payments correctly.

## Task 24: Feilsøking i hovedbok (General ledger error correction)

**Count:** 9

**Fields (DuplicateVoucherError):** `account`, `amount`

**Fields (IncorrectAmountError):** `account`, `posted_amount`, `correct_amount`

**Fields (LedgerErrorCorrection):** `wrong_account_error`, `duplicate_voucher_error`, `missing_vat_line_error`, `incorrect_amount_error`

**Fields (MissingVatLineError):** `account`, `amount_excl_vat`

**Fields (WrongAccountError):** `wrong_account`, `correct_account`, `amount`

- Me har oppdaga feil i hovudboka for januar og februar 2026. Gå gjennom alle bilag og finn dei 4 feila: ei postering på feil konto (konto 6500 brukt i staden for 6540, beløp 3450 kr), eit duplikat bilag (konto 6540, beløp 3700 kr), ei manglande MVA-linje (konto 6540, beløp ekskl. 23500 kr manglar MVA på konto 2710), og eit feil beløp (konto 7300, 18600 kr bokført i staden for 11550 kr). Korriger alle feil med rette bilag.
- We have discovered errors in the general ledger for January and February 2026. Review all vouchers and find the 4 errors: a posting to the wrong account (account 7100 used instead of 7140, amount 6400 NOK), a duplicate voucher (account 7300, amount 1100 NOK), a missing VAT line (account 6500, amount excl. 19350 NOK missing VAT on account 2710), and an incorrect amount (account 6540, 20500 NOK posted instead of 12150 NOK). Correct all errors with appropriate correction vouchers.
- Me har oppdaga feil i hovudboka for januar og februar 2026. Gå gjennom alle bilag og finn dei 4 feila: ei postering på feil konto (konto 6540 brukt i staden for 6860, beløp 4600 kr), eit duplikat bilag (konto 6860, beløp 4150 kr), ei manglande MVA-linje (konto 7000, beløp ekskl. 24100 kr manglar MVA på konto 2710), og eit feil beløp (konto 6300, 21800 kr bokført i staden for 16250 kr). Korriger alle feil med rette bilag.
- Hemos descubierto errores en el libro mayor de enero y febrero de 2026. Revise todos los comprobantes y encuentre los 4 errores: un asiento en la cuenta incorrecta (cuenta 6860 usada en lugar de 6590, importe 2200 NOK), un comprobante duplicado (cuenta 6540, importe 2200 NOK), una línea de IVA faltante (cuenta 6590, importe sin IVA 19900 NOK falta IVA en cuenta 2710), y un importe incorrecto (cuenta 6300, 19650 NOK registrado en lugar de 16550 NOK). Corrija todos los errores con asientos correctivos.
- We have discovered errors in the general ledger for January and February 2026. Review all vouchers and find the 4 errors: a posting to the wrong account (account 7300 used instead of 7000, amount 1650 NOK), a duplicate voucher (account 6500, amount 3100 NOK), a missing VAT line (account 6540, amount excl. 20950 NOK missing VAT on account 2710), and an incorrect amount (account 6340, 18100 NOK posted instead of 6050 NOK). Correct all errors with appropriate correction vouchers.
- Vi har oppdaget feil i hovedboken for januar og februar 2026. Gå gjennom alle bilag og finn de 4 feilene: en postering på feil konto (konto 7140 brukt i stedet for 7100, beløp 5850 kr), et duplisert bilag (konto 7300, beløp 1200 kr), en manglende MVA-linje (konto 6540, beløp ekskl. 13000 kr mangler MVA på konto 2710), og et feil beløp (konto 7100, 19050 kr bokført i stedet for 7100 kr). Korriger alle feil med riktige bilag.
- Descobrimos erros no livro razão de janeiro e fevereiro de 2026. Revise todos os vouchers e encontre os 4 erros: um lançamento na conta errada (conta 7300 usada em vez de 7000, valor 7800 NOK), um voucher duplicado (conta 6860, valor 3500 NOK), uma linha de IVA em falta (conta 6500, valor sem IVA 18350 NOK falta IVA na conta 2710), e um valor incorreto (conta 7300, 15000 NOK registado em vez de 10050 NOK). Corrija todos os erros com lançamentos corretivos.
- Wir haben Fehler im Hauptbuch für Januar und Februar 2026 entdeckt. Überprüfen Sie alle Belege und finden Sie die 4 Fehler: eine Buchung auf das falsche Konto (Konto 6540 statt 6860, Betrag 3150 NOK), ein doppelter Beleg (Konto 6500, Betrag 1500 NOK), eine fehlende MwSt.-Zeile (Konto 6300, Betrag ohne MwSt. 22350 NOK, fehlende MwSt. auf Konto 2710), und ein falscher Betrag (Konto 6540, 11400 NOK gebucht statt 8550 NOK). Korrigieren Sie alle Fehler mit entsprechenden Korrekturbuchungen.
- Descobrimos erros no livro razão de janeiro e fevereiro de 2026. Revise todos os vouchers e encontre os 4 erros: um lançamento na conta errada (conta 7140 usada em vez de 7100, valor 7500 NOK), um voucher duplicado (conta 6540, valor 1000 NOK), uma linha de IVA em falta (conta 4500, valor sem IVA 21500 NOK falta IVA na conta 2710), e um valor incorreto (conta 6860, 17250 NOK registado em vez de 6000 NOK). Corrija todos os erros com lançamentos corretivos.

## Task 25: Forfallen faktura med purregebyr og delbetaling (Overdue invoice + reminder fee + partial payment)

**Count:** 10

**Fields (OverdueReminder):** `reminder_fee`, `partial_payment_amount`

- Einer Ihrer Kunden hat eine uberfallige Rechnung. Finden Sie die uberfallige Rechnung und buchen Sie eine Mahngebuhr von 70 NOK. Soll Forderungen (1500), Haben Mahngebuhren (3400). Erstellen Sie außerdem eine Rechnung über die Mahngebühr an den Kunden und senden Sie diese. Registrieren Sie zusätzlich eine Teilzahlung von 5000 NOK auf der überfälligen Rechnung.
- Ein av kundane dine har ein forfallen faktura. Finn den forfalne fakturaen og bokfor eit purregebyr pa 35 kr. Debet kundefordringar (1500), kredit purregebyr (3400). Opprett også ein faktura for purregebyret til kunden og send den. Registrer i tillegg ei delbetaling på 5000 kr på den forfalne fakturaen.
- Uno de sus clientes tiene una factura vencida. Encuentre la factura vencida y registre un cargo por recordatorio de 60 NOK. Debito cuentas por cobrar (1500), credito ingresos por recordatorio (3400). También cree una factura por la tarifa de recordatorio al cliente y envíela. Además, registre un pago parcial de 5000 NOK en la factura vencida.
- Uno de sus clientes tiene una factura vencida. Encuentre la factura vencida y registre un cargo por recordatorio de 35 NOK. Debito cuentas por cobrar (1500), credito ingresos por recordatorio (3400). También cree una factura por la tarifa de recordatorio al cliente y envíela. Además, registre un pago parcial de 5000 NOK en la factura vencida.
- En av kundene dine har en forfalt faktura. Finn den forfalte fakturaen og bokfor et purregebyr pa 40 kr. Debet kundefordringer (1500), kredit purregebyr (3400). Opprett også en faktura for purregebyret til kunden og send den. Registrer i tillegg en delbetaling på 5000 kr på den forfalte fakturaen.
- L'un de vos clients a une facture en retard. Trouvez la facture en retard et enregistrez des frais de rappel de 70 NOK. Debit creances clients (1500), credit revenus de rappel (3400). Créez également une facture pour les frais de rappel au client et envoyez-la. De plus, enregistrez un paiement partiel de 5000 NOK sur la facture en retard.
- One of your customers has an overdue invoice. Find the overdue invoice and post a reminder fee of 70 NOK. Debit accounts receivable (1500), credit reminder fees (3400). Also create an invoice for the reminder fee to the customer and send it. Additionally, register a partial payment of 5000 NOK on the overdue invoice.
- Uno de sus clientes tiene una factura vencida. Encuentre la factura vencida y registre un cargo por recordatorio de 65 NOK. Debito cuentas por cobrar (1500), credito ingresos por recordatorio (3400). También cree una factura por la tarifa de recordatorio al cliente y envíela. Además, registre un pago parcial de 5000 NOK en la factura vencida.
- Uno de sus clientes tiene una factura vencida. Encuentre la factura vencida y registre un cargo por recordatorio de 55 NOK. Debito cuentas por cobrar (1500), credito ingresos por recordatorio (3400). También cree una factura por la tarifa de recordatorio al cliente y envíela. Además, registre un pago parcial de 5000 NOK en la factura vencida.
- Um dos seus clientes tem uma fatura vencida. Encontre a fatura vencida e registe uma taxa de lembrete de 40 NOK. Debito contas a receber (1500), credito receitas de lembrete (3400). Também crie uma fatura para a taxa de lembrete ao cliente e envie-a. Além disso, registe um pagamento parcial de 5000 NOK na fatura vencida.

## Task 26: Periodeavslutning / månedsavslutning (Monthly closing)

**Count:** 7

**Fields (MonthlyClosing):** `accrual_amount_per_month`, `accrual_source_account`, `asset_cost`, `useful_life_years`, `depreciation_account`

- Führen Sie den Monatsabschluss für März 2026 durch. Buchen Sie die Rechnungsabgrenzung (4200 NOK pro Monat von Konto 1700 auf Aufwand). Erfassen Sie die monatliche Abschreibung für eine Anlage mit Anschaffungskosten 64000 NOK und Nutzungsdauer 3 Jahre (lineare Abschreibung auf Konto 6030). Überprüfen Sie, ob die Saldenbilanz null ergibt. Buchen Sie außerdem eine Gehaltsrückstellung (Soll Gehaltsaufwand Konto 5000, Haben aufgelaufene Gehälter Konto 2900).
- Realize o encerramento mensal de março de 2026. Registe a reversão de acréscimos (9700 NOK por mês da conta 1710 para despesa). Registe a depreciação mensal de um ativo fixo com custo de aquisição 117300 NOK e vida útil 3 anos (depreciação linear para conta 6020). Verifique se o balancete está a zero. Registe também uma provisão salarial (débito conta de despesas salariais 5000, crédito conta de salários acumulados 2900).
- Führen Sie den Monatsabschluss für März 2026 durch. Buchen Sie die Rechnungsabgrenzung (2200 NOK pro Monat von Konto 1720 auf Aufwand). Erfassen Sie die monatliche Abschreibung für eine Anlage mit Anschaffungskosten 291700 NOK und Nutzungsdauer 5 Jahre (lineare Abschreibung auf Konto 6020). Überprüfen Sie, ob die Saldenbilanz null ergibt. Buchen Sie außerdem eine Gehaltsrückstellung (Soll Gehaltsaufwand Konto 5000, Haben aufgelaufene Gehälter Konto 2900).
- Realize o encerramento mensal de março de 2026. Registe a reversão de acréscimos (9000 NOK por mês da conta 1700 para despesa). Registe a depreciação mensal de um ativo fixo com custo de aquisição 289500 NOK e vida útil 9 anos (depreciação linear para conta 6030). Verifique se o balancete está a zero. Registe também uma provisão salarial (débito conta de despesas salariais 5000, crédito conta de salários acumulados 2900).
- Führen Sie den Monatsabschluss für März 2026 durch. Buchen Sie die Rechnungsabgrenzung (3500 NOK pro Monat von Konto 1710 auf Aufwand). Erfassen Sie die monatliche Abschreibung für eine Anlage mit Anschaffungskosten 79150 NOK und Nutzungsdauer 6 Jahre (lineare Abschreibung auf Konto 6020). Überprüfen Sie, ob die Saldenbilanz null ergibt. Buchen Sie außerdem eine Gehaltsrückstellung (Soll Gehaltsaufwand Konto 5000, Haben aufgelaufene Gehälter Konto 2900).
- Führen Sie den Monatsabschluss für März 2026 durch. Buchen Sie die Rechnungsabgrenzung (6150 NOK pro Monat von Konto 1710 auf Aufwand). Erfassen Sie die monatliche Abschreibung für eine Anlage mit Anschaffungskosten 181400 NOK und Nutzungsdauer 3 Jahre (lineare Abschreibung auf Konto 6020). Überprüfen Sie, ob die Saldenbilanz null ergibt. Buchen Sie außerdem eine Gehaltsrückstellung (Soll Gehaltsaufwand Konto 5000, Haben aufgelaufene Gehälter Konto 2900).
- Führen Sie den Monatsabschluss für März 2026 durch. Buchen Sie die Rechnungsabgrenzung (3400 NOK pro Monat von Konto 1700 auf Aufwand). Erfassen Sie die monatliche Abschreibung für eine Anlage mit Anschaffungskosten 289700 NOK und Nutzungsdauer 7 Jahre (lineare Abschreibung auf Konto 6020). Überprüfen Sie, ob die Saldenbilanz null ergibt. Buchen Sie außerdem eine Gehaltsrückstellung (Soll Gehaltsaufwand Konto 5000, Haben aufgelaufene Gehälter Konto 2900).

## Task 27: Valutafaktura med agio/disagio (Currency invoice with exchange rate difference)

**Count:** 10

**Fields (CurrencyInvoice):** `eur_amount`, `customer_name`, `org_number`, `original_rate`, `payment_rate`, `exchange_rate_type`

- Enviámos uma fatura de 8676 EUR ao Cascata Lda (org. nº 967770892) quando a taxa de câmbio era 11.62 NOK/EUR. O cliente pagou agora, mas a taxa é 10.89 NOK/EUR. Registe o pagamento e lance a diferença cambial (disagio) na conta correta.
- Enviámos uma fatura de 2336 EUR ao Estrela Lda (org. nº 808808773) quando a taxa de câmbio era 11.17 NOK/EUR. O cliente pagou agora, mas a taxa é 12.13 NOK/EUR. Registe o pagamento e lance a diferença cambial (agio) na conta correta.
- Nous avons envoyé une facture de 9273 EUR à Montagne SARL (nº org. 941382142) lorsque le taux de change était de 10.34 NOK/EUR. Le client a maintenant payé, mais le taux est de 10.83 NOK/EUR. Enregistrez le paiement et comptabilisez l'écart de change (agio) sur le bon compte.
- Wir haben eine Rechnung über 19629 EUR an Waldstein GmbH (Org.-Nr. 919320044) gesendet, als der Wechselkurs 11.14 NOK/EUR betrug. Der Kunde hat nun bezahlt, aber der Kurs liegt bei 10.74 NOK/EUR. Erfassen Sie die Zahlung und buchen Sie die Wechselkursdifferenz (disagio) auf das korrekte Konto.
- Enviamos una factura por 9487 EUR a Estrella SL (org. nº 834293692) cuando el tipo de cambio era 11.54 NOK/EUR. El cliente ha pagado ahora, pero el tipo es 10.95 NOK/EUR. Registre el pago y contabilice la diferencia de tipo de cambio (disagio) en la cuenta correcta.
- Wir haben eine Rechnung über 8077 EUR an Bergwerk GmbH (Org.-Nr. 863352010) gesendet, als der Wechselkurs 10.91 NOK/EUR betrug. Der Kunde hat nun bezahlt, aber der Kurs liegt bei 10.29 NOK/EUR. Erfassen Sie die Zahlung und buchen Sie die Wechselkursdifferenz (disagio) auf das korrekte Konto.
- Me sende ein faktura på 12301 EUR til Bølgekraft AS (org.nr 830993940) då kursen var 10.83 NOK/EUR. Kunden har no betalt, men kursen er 11.83 NOK/EUR. Registrer betalinga og bokfør valutadifferansen (agio) på rett konto.
- Me sende ein faktura på 9364 EUR til Elvdal AS (org.nr 808826054) då kursen var 11.65 NOK/EUR. Kunden har no betalt, men kursen er 12.35 NOK/EUR. Registrer betalinga og bokfør valutadifferansen (agio) på rett konto.
- Nous avons envoyé une facture de 6224 EUR à Rivière SARL (nº org. 886727127) lorsque le taux de change était de 10.90 NOK/EUR. Le client a maintenant payé, mais le taux est de 10.33 NOK/EUR. Enregistrez le paiement et comptabilisez l'écart de change (disagio) sur le bon compte.
- Enviámos uma fatura de 10765 EUR ao Estrela Lda (org. nº 950948086) quando a taxa de câmbio era 11.30 NOK/EUR. O cliente pagou agora, mas a taxa é 10.78 NOK/EUR. Registe o pagamento e lance a diferença cambial (disagio) na conta correta.

## Task 28: Resultatanalyse, kostnadsøkning (Cost increase analysis + create internal projects)

**Count:** 5

- Los costos totales aumentaron significativamente de enero a febrero de 2026. Analice el libro mayor e identifique las tres cuentas de gastos con el mayor incremento en monto. Cree un proyecto interno para cada una de las tres cuentas con el nombre de la cuenta. También cree una actividad para cada proyecto.
- Les coûts totaux ont augmenté de manière significative de janvier à février 2026. Analysez le grand livre et identifiez les trois comptes de charges avec la plus forte augmentation. Créez un projet interne pour chacun des trois comptes avec le nom du compte. Créez également une activité pour chaque projet.
- Os custos totais aumentaram significativamente de janeiro a fevereiro de 2026. Analise o livro razão e identifique as três contas de despesa com o maior aumento em valor. Crie um projeto interno para cada uma das três contas com o nome da conta. Também crie uma atividade para cada projeto.
- Totalkostnadene auka monaleg frå januar til februar 2026. Analyser hovudboka og finn dei tre kostnadskontoane med størst auke i beløp. Opprett eit internt prosjekt for kvar av dei tre kontoane med kontoens namn. Opprett også ein aktivitet for kvart prosjekt.
- Die Gesamtkosten sind von Januar bis Februar 2026 deutlich gestiegen. Analysieren Sie das Hauptbuch und identifizieren Sie die drei Aufwandskonten mit dem größten Anstieg. Erstellen Sie für jedes der drei Konten ein internes Projekt mit dem Kontonamen. Erstellen Sie außerdem eine Aktivität für jedes Projekt.

## Task 29: Komplett prosjektsyklus med avvik (Complete project lifecycle)

**Count:** 9

**Fields (CompleteProjectLifecycle):** `project_name`, `customer_name`, `customer_org_number`, `budget`, `project_manager`, `consultant`, `supplier_cost`, `supplier_name`, `supplier_org_number`

**Fields (TeamMember):** `first_name`, `last_name`, `email`, `hours`

- Gjennomfør heile prosjektsyklusen for 'Dataplattform Strandvik' (Strandvik AS, org.nr 982465958): 1) Prosjektet har budsjett 412400 kr. 2) Registrer timar: Olav Stølsvik (prosjektleiar, olav.stlsvik@example.org) 29 timar og Jorunn Eide (konsulent, jorunn.eide@example.org) 100 timar. 3) Registrer leverandørkostnad 49600 kr frå Skogheim AS (org.nr 950515430). 4) Opprett kundefaktura for prosjektet.
- Gjennomfør hele prosjektsyklusen for 'ERP-implementering Snøhetta' (Snøhetta AS, org.nr 954447499): 1) Prosjektet har budsjett 431600 kr. 2) Registrer timer: Sigurd Johansen (prosjektleder, sigurd.johansen@example.org) 47 timer og Erik Haugen (konsulent, erik.haugen@example.org) 46 timer. 3) Registrer leverandørkostnad 95050 kr fra Nordhav AS (org.nr 957929974). 4) Opprett kundefaktura for prosjektet.
- Execute the complete project lifecycle for 'Data Platform Ridgepoint' (Ridgepoint Ltd, org no. 808812096): 1) The project has a budget of 316550 NOK. 2) Log time: Daniel Harris (project manager, daniel.harris@example.org) 54 hours and Grace Johnson (consultant, grace.johnson@example.org) 78 hours. 3) Register supplier cost of 58200 NOK from Ironbridge Ltd (org no. 814796019). 4) Create a customer invoice for the project.
- Ejecute el ciclo de vida completo del proyecto 'Migración Cloud Montaña' (Montaña SL, org. nº 903805021): 1) El proyecto tiene un presupuesto de 219800 NOK. 2) Registre horas: Ana Rodríguez (director de proyecto, ana.rodriguez@example.org) 36 horas y Isabel García (consultor, isabel.garcia@example.org) 69 horas. 3) Registre costo de proveedor de 46050 NOK de Luna SL (org. nº 988327581). 4) Cree una factura al cliente por el proyecto.
- Gjennomfør heile prosjektsyklusen for 'Dataplattform Sjøbris' (Sjøbris AS, org.nr 868541946): 1) Prosjektet har budsjett 361050 kr. 2) Registrer timar: Eirik Stølsvik (prosjektleiar, eirik.stlsvik@example.org) 41 timar og Geir Aasen (konsulent, geir.aasen@example.org) 122 timar. 3) Registrer leverandørkostnad 23500 kr frå Elvdal AS (org.nr 964202133). 4) Opprett kundefaktura for prosjektet.
- Execute the complete project lifecycle for 'Cloud Migration Northwave' (Northwave Ltd, org no. 932075482): 1) The project has a budget of 396900 NOK. 2) Log time: Samuel Brown (project manager, samuel.brown@example.org) 74 hours and Sarah Lewis (consultant, sarah.lewis@example.org) 85 hours. 3) Register supplier cost of 56750 NOK from Clearwater Ltd (org no. 889264985). 4) Create a customer invoice for the project.
- Gjennomfør hele prosjektsyklusen for 'ERP-implementering Bergvik' (Bergvik AS, org.nr 886943407): 1) Prosjektet har budsjett 400950 kr. 2) Registrer timer: Ragnhild Strand (prosjektleder, ragnhild.strand@example.org) 35 timer og Silje Bakken (konsulent, silje.bakken@example.org) 79 timer. 3) Registrer leverandørkostnad 94350 kr fra Brattli AS (org.nr 810297891). 4) Opprett kundefaktura for prosjektet.
- Gjennomfør heile prosjektsyklusen for 'Dataplattform Skogheim' (Skogheim AS, org.nr 841795067): 1) Prosjektet har budsjett 258650 kr. 2) Registrer timar: Torbjørn Brekke (prosjektleiar, torbjrn.brekke@example.org) 64 timar og Arne Kvamme (konsulent, arne.kvamme@example.org) 87 timar. 3) Registrer leverandørkostnad 77950 kr frå Nordlys AS (org.nr 894689668). 4) Opprett kundefaktura for prosjektet.
- Exécutez le cycle de vie complet du projet 'Portail Numérique Étoile' (Étoile SARL, nº org. 834437961) : 1) Le projet a un budget de 383650 NOK. 2) Enregistrez le temps : Jade Martin (chef de projet, jade.martin@example.org) 53 heures et Louis Robert (consultant, louis.robert@example.org) 56 heures. 3) Enregistrez un coût fournisseur de 90100 NOK de Montagne SARL (nº org. 891743882). 4) Créez une facture client pour le projet.

## Task 30: Årsoppgjør (Simplified year-end closing)

**Count:** 8

**Fields (DepreciationAsset):** `asset_name`, `cost`, `useful_life_years`, `asset_account`

**Fields (YearEndClosing):** `assets`, `prepaid_expense_total`

- Realize o encerramento anual simplificado de 2025: 1) Calcule e registe a depreciação anual de três ativos: Kontormaskiner (301750 NOK, 10 anos lineares, conta 1200), IT-utstyr (304100 NOK, 3 anos, conta 1210), Programvare (322800 NOK, 8 anos, conta 1250). Use conta 6010 para despesa de depreciação e 1209 para depreciação acumulada. 2) Reverta despesas antecipadas (total 27150 NOK na conta 1700). 3) Calcule e registe a provisão fiscal (22 % do resultado tributável) na conta 8700/2920. Registe cada depreciação como um lançamento separado.
- Führen Sie den vereinfachten Jahresabschluss für 2025 durch: 1) Berechnen und buchen Sie die jährliche Abschreibung für drei Anlagen: Programvare (419000 NOK, 6 Jahre linear, Konto 1250), Kontormaskiner (374150 NOK, 8 Jahre, Konto 1200), Kjøretøy (64300 NOK, 9 Jahre, Konto 1230). Verwenden Sie Konto 6010 für Abschreibungsaufwand und 1209 für kumulierte Abschreibungen. 2) Lösen Sie vorausbezahlte Aufwendungen auf (insgesamt 51000 NOK auf Konto 1700). 3) Berechnen und buchen Sie die Steuerrückstellung (22 % des steuerpflichtigen Gewinns) auf Konto 8700/2920. Buchen Sie jede Abschreibung als separaten Beleg.
- Perform simplified year-end closing for 2025: 1) Calculate and post annual depreciation for three assets: Kontormaskiner (138450 NOK, 6 years straight-line, account 1200), Programvare (280000 NOK, 9 years, account 1250), Inventar (484650 NOK, 8 years, account 1240). Use account 6010 for depreciation expense and 1209 for accumulated depreciation. 2) Reverse prepaid expenses (total 52250 NOK on account 1700). 3) Calculate and post tax provision (22% of taxable profit) on account 8700/2920. Post each depreciation as a separate voucher.
- Effectuez la clôture annuelle simplifiée pour 2025 : 1) Calculez et comptabilisez l'amortissement annuel de trois immobilisations : Programvare (111950 NOK, 9 ans linéaire, compte 1250), Kontormaskiner (351450 NOK, 9 ans, compte 1200), Inventar (418800 NOK, 10 ans, compte 1240). Utilisez le compte 6010 pour les charges d'amortissement et 1209 pour les amortissements cumulés. 2) Extournez les charges constatées d'avance (total 79750 NOK au compte 1700). 3) Calculez et comptabilisez la provision d'impôt (22 % du bénéfice imposable) sur compte 8700/2920. Comptabilisez chaque amortissement comme une pièce comptable séparée.
- Utfør forenklet årsoppgjør for 2025: 1) Beregn og bokfør årlige avskrivninger for tre eiendeler: Inventar (468300 kr, 10 år lineært, konto 1240), Kontormaskiner (149750 kr, 10 år, konto 1200), Kjøretøy (412950 kr, 9 år, konto 1230). Bruk konto 6010 for avskrivningskostnad og 1209 for akkumulerte avskrivninger. 2) Reverser forskuddsbetalte kostnader (totalt 47850 kr på konto 1700). 3) Beregn og bokfør skattekostnad (22 % av skattbart resultat) på konto 8700/2920. Bokfør hver avskrivning som et eget bilag.
- Führen Sie den vereinfachten Jahresabschluss für 2025 durch: 1) Berechnen und buchen Sie die jährliche Abschreibung für drei Anlagen: IT-utstyr (229850 NOK, 10 Jahre linear, Konto 1210), Kontormaskiner (146050 NOK, 8 Jahre, Konto 1200), Kjøretøy (311650 NOK, 6 Jahre, Konto 1230). Verwenden Sie Konto 6010 für Abschreibungsaufwand und 1209 für kumulierte Abschreibungen. 2) Lösen Sie vorausbezahlte Aufwendungen auf (insgesamt 27650 NOK auf Konto 1700). 3) Berechnen und buchen Sie die Steuerrückstellung (22 % des steuerpflichtigen Gewinns) auf Konto 8700/2920. Buchen Sie jede Abschreibung als separaten Beleg.
- Realice el cierre anual simplificado de 2025: 1) Calcule y contabilice la depreciación anual de tres activos: Programvare (359300 NOK, 4 años lineales, cuenta 1250), IT-utstyr (444050 NOK, 6 años, cuenta 1210), Kjøretøy (344650 NOK, 3 años, cuenta 1230). Use cuenta 6010 para gasto de depreciación y 1209 para depreciación acumulada. 2) Revierta gastos prepagados (total 36150 NOK en cuenta 1700). 3) Calcule y contabilice la provisión de impuestos (22 % del resultado imponible) en cuenta 8700/2920. Registre cada depreciación como un comprobante separado.
- Perform simplified year-end closing for 2025: 1) Calculate and post annual depreciation for three assets: IT-utstyr (204150 NOK, 4 years straight-line, account 1210), Inventar (237550 NOK, 8 years, account 1240), Programvare (307500 NOK, 4 years, account 1250). Use account 6010 for depreciation expense and 1209 for accumulated depreciation. 2) Reverse prepaid expenses (total 44300 NOK on account 1700). 3) Calculate and post tax provision (22% of taxable profit) on account 8700/2920. Post each depreciation as a separate voucher.
