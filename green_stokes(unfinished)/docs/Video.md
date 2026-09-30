# Ordine delle scene

Dodici segmenti separati, in ordine narrativo. Le durate indicano i render locali a 1920×1080 e 60 fps, senza audio. I file video non sono inclusi nel repository: si possono rigenerare con `python3 render.py --all` dalla cartella del progetto.

| Ordine | Scena | Durata |
|---|---|---:|
| 01 | `VecFieldWithSteam` — Campo vettoriale e rotore | 47.0 s |
| 02 | `VecFieldWithSteamContinua` — Rotore visto in 3D | 32.0 s |
| 03 | `Hypotesys` — Ipotesi di regolarità | 13.0 s |
| 04 | `SumDotProducts` — Prodotto scalare lungo la curva | 23.3 s |
| 05 | `CirculationIntegral` — Dalla somma alla circuitazione | 22.8 s |
| 06 | `GreenAnalytic` — Dimostrazione sul quadrato | 55.0 s |
| 07 | `LocalLimit` — Limite sul quadrato piccolo | 21.5 s |
| 08 | `TwoSquares` — Cancellazione fra due quadrati | 20.2 s |
| 09 | `GreenGlobal` — Teorema di Green globale | 36.9 s |
| 10 | `StokesSurface` — Teorema di Stokes sulla superficie | 47.7 s |
| 11 | `FundamentalTheorem` — Teorema fondamentale del calcolo | 27.9 s |
| 12 | `GeneralizedStokes` — Teorema di Stokes generalizzato | 32.6 s |

Per modificare le scene, aprire `main.py` e il modulo indicato in ciascuna classe. Per rigenerarle, usare `render.py`.

[Guida del progetto](../README.md)
