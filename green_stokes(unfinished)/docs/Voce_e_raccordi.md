# Voce e raccordi

Testo italiano proposto per accompagnare le nuove scene. Non è una registrazione e non è una sincronizzazione definitiva: i video includono pause di lettura modificabili nei sorgenti. I riferimenti temporali seguenti sono indicativi e locali al segmento.

## CirculationIntegral — raccordo dopo SumDotProducts

«Ogni prodotto scalare descrive un contributo locale. Per sommarli correttamente dobbiamo includere anche il piccolo incremento del parametro. Aumentando il numero dei campioni, la somma diventa un integrale: è la circuitazione del campo lungo la curva chiusa.»

0–4 s: continuità con curva, punto e vettori del segmento precedente; 4–12 s: contributo e somma; 12–18 s: raffinamento e integrale; ultimi secondi: formula ferma.

## LocalLimit — dopo la derivazione sul quadrato

«Con lo stesso campo, fissiamo questo punto e restringiamo il quadrato. La circuitazione divisa per l'area vale tre più due volte il lato. Il rotore nel punto vale tre. Più il lato diminuisce, più la differenza scompare: il rotore è il limite della circuitazione per unità di area.»

Il campo è F=(-y²,x²); il quadrato ha vertice inferiore sinistro p=(1,1/2) e lato h. La densità esatta è 3+2h. Non si usa il centro del quadrato come punto di valutazione: per questo esempio la differenza è effettivamente non nulla e converge a zero.

## TwoSquares — cancellazione del lato comune

«Nei due quadrati orientati allo stesso modo, il lato condiviso viene percorso in versi opposti. Sono due integrali sullo stesso tratto, con segno contrario: si cancellano esattamente. Rimane soltanto il percorso lungo il bordo del rettangolo formato dai due quadrati.»

La versione corretta è già integrata nel progetto e nel video `TwoSquares.mp4`. Mostra esplicitamente la coppia corretta di integrali Q(x+Δx,t).

## GreenGlobal — chiusura di Green

«La stessa cancellazione vale per qualunque insieme di quadratini adiacenti. Ogni lato interno compare due volte, con versi opposti; soltanto il bordo esterno sopravvive. Ora avviciniamo questa costruzione a una regione curva. Riducendo il lato dei quadratini, il contorno a gradini si avvicina al bordo della regione. Su ciascuna cella, la circuitazione è approssimata dal rotore per l'area. Nel limite, la somma diventa un integrale doppio. La circuitazione sul bordo e l'integrale del rotore nell'interno coincidono: questo è il teorema di Green.»

0–12 s: cancellazione esatta su sei celle; 12–25 s: regione curva e tre risoluzioni; 25–34 s: limite e formula; ultimi secondi: la curva ritorna al centro, pronta per Stokes.

Le ipotesi restano quelle del teorema: regione con bordo sufficientemente regolare, orientato positivamente, e campo C¹ in un intorno della regione.

## StokesSurface — superficie curva

«Possiamo ripetere il ragionamento su una superficie curva. Scegliamo una superficie orientabile e una normale coerente, che ora cambia direzione da punto a punto. Il rotore è un vettore: per ogni piccolo tassello conta la sua componente lungo la normale, moltiplicata per l'area del tassello. Anche qui, due tasselli vicini percorrono il loro lato comune in versi opposti. Sommando su tutta la superficie, i contributi interni si cancellano e resta il bordo. Nel limite otteniamo Stokes: la circuitazione sul bordo è uguale al flusso del rotore attraverso la superficie. Il verso del bordo deve essere compatibile con quello della normale.»

0–6 s: dal piano alla superficie; 6–13 s: normali; 13–21 s: proiezione del rotore e contributo locale; 21–29 s: due tasselli; 29–36 s: cancellazione globale; 36 s–fine: formula e bordo.

Il campo deve essere C¹ in un intorno della superficie. I vettori nella dimostrazione locale sono illustrativi: la clip non attribuisce loro una scala numerica o una particolare legge del campo. Le normali, il reticolo, le adiacenze e la proiezione sono calcolati dalla geometria della superficie.

## FundamentalTheorem — caso 1D

«In una dimensione la regione è un intervallo e il suo bordo è formato dagli estremi. Su ogni piccolo segmento consideriamo la differenza fra il valore finale e quello iniziale. Sommando, i valori nei punti interni si cancellano a coppie. Restano il valore in b e l'opposto del valore in a. Passando alla derivata e al limite, ritroviamo il teorema fondamentale del calcolo integrale.»

Non dire che in 1D esiste un “rotore” scalare analogo a quello piano. Qui il cambiamento locale è la derivata; il collegamento rigoroso è con Stokes generalizzato.

## GeneralizedStokes — sintesi e conclusione

«Un intervallo, una regione piana, una superficie curva: in tutti questi casi ritroviamo la stessa struttura. Il linguaggio delle forme differenziali permette di scriverla in un'unica formula. Omega rappresenta la quantità che integriamo sul bordo; d omega è la sua derivata esterna, integrata nella regione. Su una varietà orientata, con le opportune condizioni di regolarità, i due integrali coincidono. È il teorema di Stokes nella sua forma generale.»

La presentazione assume il caso compatto regolare con bordo e una forma sufficientemente regolare. La formula non afferma una generica conservazione fisica dell'informazione.

## Precisazione alla voce già scritta per Hypotesys

Sostituire l'affermazione che entrambi gli esempi sono discontinui con:

«Il primo campo non è continuo nell'origine. Il secondo è continuo, ma presenta una componente non derivabile lungo x uguale a zero. Sono due modi diversi in cui può venire meno la regolarità richiesta.»

I render originali non contengono audio; questa modifica va applicata al parlato, quando viene registrato o montato.
