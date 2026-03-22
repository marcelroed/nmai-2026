You are given a few variants of the same accounting task. Your job is to correctly identify the required api endpoints to achieve the described task, while using as few write calls (POST, PUT, DELETE, PATCH) as possible.  GET requests are not counted — read as much as you need to understand the data.

Assume that in the first step, we can parse the natural language prompt into any arbitrary pydantic basemodel. You do not need to implement that method, but assume the data you use is coming from this basemodel. You still need to define the shape of the BaseModel as well as what (nested) fields it contains.

Use the openapi spec and make sure you look for methods that can both improve correctness and that can allow us to achieve everything in fewer number of calls.

In your answer, you should both provide a written response with which tripletex api calls you need and what you would send to them, as well as code which both defines the pydantic basemodel and the request apis.

Read through the problem instructions at https://app.ainm.no/docs/tripletex/overview, https://app.ainm.no/docs/tripletex/sandbox, https://app.ainm.no/docs/tripletex/endpoint, https://app.ainm.no/docs/tripletex/scoring and https://app.ainm.no/docs/tripletex/examples

Task:

TASK 18 - TIER 2: Reverser betaling (bankretur)

Example prompts:

"Betalinga frå Dalheim AS (org.nr 979527578) for fakturaen \"Systemutvikling\" (14550 kr ekskl. MVA) vart returnert av banken. Reverser betalinga slik at fakturaen igjen viser uteståande beløp.",
"Die Zahlung von Grünfeld GmbH (Org.-Nr. 808603152) für die Rechnung \"Netzwerkdienst\" (44300 NOK ohne MwSt.) wurde von der Bank zurückgebucht. Stornieren Sie die Zahlung, damit die Rechnung wieder den offenen Betrag anzeigt.",
"El pago de Dorada SL (org. nº 849807021) por la factura \"Mantenimiento\" (8900 NOK sin IVA) fue devuelto por el banco. Revierta el pago para que la factura vuelva a mostrar el importe pendiente.",
"El pago de Olivares SL (org. nº 921296819) por la factura \"Almacenamiento en la nube\" (46300 NOK sin IVA) fue devuelto por el banco. Revierta el pago para que la factura vuelva a mostrar el importe pendiente.",
"Kunden Run46 AS (org.nr 946293100) har betalt en faktura, men betalingen ble returnert av banken. Reverser betalingen.",
"Kunden Validator AS (org.nr 999888777) har betalt en faktura, men betalingen ble returnert av banken. Reverser betalingen.",
"Kunden Validator AS (org.nr 999888777) har ein uteståande faktura. Betalinga vart returnert av banken. Reverser betalinga slik at fakturaen igjen viser uteståande beløp.",
"Le paiement de Colline SARL (nº org. 804499180) pour la facture \"Heures de conseil\" (8850 NOK HT) a été retourné par la banque. Annulez le paiement afin que la facture affiche à nouveau le montant impayé.",
"Le paiement de Lumière SARL (nº org. 937075405) pour la facture \"Conseil en données\" (39100 NOK HT) a été retourné par la banque. Annulez le paiement afin que la facture affiche à nouveau le montant impayé.",
"The payment from Blueshore Ltd (org no. 989902121) for the invoice \"Software License\" (39700 NOK excl. VAT) was returned by the bank. Reverse the payment so the invoice shows the outstanding amount again.",
"The payment from Ironbridge Ltd (org no. 966448148) for the invoice \"System Development\" (45800 NOK excl. VAT) was returned by the bank. Reverse the payment so the invoice shows the outstanding amount again."
