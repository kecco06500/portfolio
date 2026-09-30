# Green & Stokes

**Il progetto si usa da `main.py`.** Le scene sono divise per argomento in `scene/`. I video si generano localmente e non sono inclusi nel repository.

[Ordine e durate delle scene](docs/Video.md) · [Voce e raccordi](docs/Voce_e_raccordi.md)

## Dove modificare il codice

| File | Contenuto |
|---|---|
| `main.py` | Punto d’ingresso Manim; tutte le scene, con gli stessi nomi dei video |
| `scene/introduzione.py` | Campo vettoriale, flussi, rotore e ipotesi |
| `scene/green.py` | Prodotto scalare, derivazione sul quadrato e due quadrati; correzioni già applicate |
| `scene/completamento.py` | I sei segmenti nuovi, fino a Stokes generalizzato |
| `scene/stile.py` | Palette, formule, curve, reticoli e helper comuni |
| `scene/esperimenti.py` | Prove storiche e abbozzi conservati |

Le piccole classi in `main.py` rimandano al codice del modulo indicato nel commento. Questo permette di lanciare tutto dallo stesso file senza avere migliaia di righe mischiate insieme.

## Comandi rapidi

```bash
# Elenco delle scene, nell’ordine narrativo
python3 render.py --list

# Anteprima di una scena e apertura del video
python3 render.py StokesSurface --preview --open

# Le sei scene nuove in 1080p60
python3 render.py --new

# Tutte le dodici scene principali, come MP4 separati
python3 render.py --all
```

Installa le dipendenze Python con `python3 -m pip install -r requirements.txt` e configura anche le dipendenze di sistema di Manim, tra cui LaTeX per le formule. `render.py` usa l’ambiente Manim attivo o un ambiente virtuale locale, se presente. I finali finiscono in `video/1080p60/`, le anteprime in `video/480p15/`. Le anteprime non sovrascrivono i finali.

Restano validi i normali comandi Manim, con il tuo ambiente attivo:

```bash
manim -pql main.py StokesSurface
manim -qh main.py GreenAnalytic TwoSquares
```

Per renderizzare tutto usare `render.py --all`: `manim -a main.py` includerebbe anche le sei prove storiche.

## Ordine delle scene principali

1. `VecFieldWithSteam` — campo e rotore
2. `VecFieldWithSteamContinua` — rotore visto in 3D
3. `Hypotesys` — regolarità
4. `SumDotProducts` — prodotto scalare lungo la curva
5. `CirculationIntegral` — dalla somma all’integrale
6. `GreenAnalytic` — derivazione sul quadrato
7. `LocalLimit` — quadrato piccolo e limite
8. `TwoSquares` — cancellazione del lato comune
9. `GreenGlobal` — chiusura di Green
10. `StokesSurface` — superficie curva e Stokes
11. `FundamentalTheorem` — caso 1D
12. `GeneralizedStokes` — conclusione

Ogni scena produce un file separato. Non viene eseguito alcun montaggio.

## Correzioni integrate

- `GreenAnalytic`: corretti i segni degli integrali orientati C→D e D→A.
- `SumDotProducts`: corretta l’uscita della lente e sostituito il numero LaTeX rigenerato ad ogni frame con `DecimalNumber`.
- `TwoSquares`: integrata la versione con cancellazione esatta dei due contributi sul lato condiviso e bordo ∂(R₁∪R₂).
- Conservate le modifiche locali già presenti alle chiamate dei flussi, inclusa la rimozione di `n_cycles`.

I video corretti hanno ora i nomi canonici `GreenAnalytic.mp4`, `SumDotProducts.mp4` e `TwoSquares.mp4`: non serve ricordarsi suffissi come Fixed o Corrected.

Gli abbozzi `GreenVisual` e `Finale` sono conservati fra gli esperimenti: la loro continuazione completa è `GreenGlobal`. La precisazione al parlato di `Hypotesys` è in `docs/Voce_e_raccordi.md`; il copione Word non è stato modificato.

## Cartelle

- **`video/`**: render generati localmente, separati per qualità; esclusi da Git.
- **`docs/`**: ordine delle scene e indicazioni per la voce.
- **`work/`**: cache temporanea di Manim, esclusa da Git.
- **`verifiche/`, `archivio/`, `outputs/`**: materiali di lavoro locali, esclusi da Git.

## Ambiente e controlli

Il progetto usa Manim CE 0.19.2 e Cairo; le dipendenze Python sono in `requirements.txt`. Con l’ambiente Manim attivo, `python3 verifica.py` controlla orientazioni, cancellazioni, integrali e normali. I video sono senza audio e i tempi del parlato si possono regolare separatamente.
