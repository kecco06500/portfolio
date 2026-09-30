"""Green–Stokes: tutte le scene partono da questo file.

Anteprima: manim -pql main.py StokesSurface
Finale:    manim -qh main.py StokesSurface
Elenco:    python render.py --list

Il codice modificabile è nei moduli scene/. Le classi locali permettono
a Manim di trovare tutte le scene mantenendo invariati i nomi storici.
"""
from scene import introduzione as _introduzione
from scene import green as _green
from scene import completamento as _completamento
from scene import esperimenti as _esperimenti

# SCENE PRINCIPALI — ordine narrativo

class VecFieldWithSteam(_introduzione.VecFieldWithSteam):
    """01. Campo vettoriale e rotore. Modificare scene/introduzione.py."""
    pass


class VecFieldWithSteamContinua(_introduzione.VecFieldWithSteamContinua):
    """02. Rotore visto in 3D. Modificare scene/introduzione.py."""
    pass


class Hypotesys(_introduzione.Hypotesys):
    """03. Ipotesi di regolarità. Modificare scene/introduzione.py."""
    pass


class SumDotProducts(_green.SumDotProducts):
    """04. Prodotto scalare lungo la curva. Modificare scene/green.py."""
    pass


class CirculationIntegral(_completamento.CirculationIntegral):
    """05. Dalla somma alla circuitazione. Modificare scene/completamento.py."""
    pass


class GreenAnalytic(_green.GreenAnalytic):
    """06. Dimostrazione sul quadrato. Modificare scene/green.py."""
    pass


class LocalLimit(_completamento.LocalLimit):
    """07. Limite sul quadrato piccolo. Modificare scene/completamento.py."""
    pass


class TwoSquares(_green.TwoSquares):
    """08. Cancellazione fra due quadrati. Modificare scene/green.py."""
    pass


class GreenGlobal(_completamento.GreenGlobal):
    """09. Teorema di Green globale. Modificare scene/completamento.py."""
    pass


class StokesSurface(_completamento.StokesSurface):
    """10. Teorema di Stokes sulla superficie. Modificare scene/completamento.py."""
    pass


class FundamentalTheorem(_completamento.FundamentalTheorem):
    """11. Teorema fondamentale del calcolo. Modificare scene/completamento.py."""
    pass


class GeneralizedStokes(_completamento.GeneralizedStokes):
    """12. Teorema di Stokes generalizzato. Modificare scene/completamento.py."""
    pass


# PROVE STORICHE — accessibili con i vecchi nomi, escluse da render.py --all

class VecFieldWithMass(_esperimenti.VecFieldWithMass):
    """Prova storica; modificare scene/esperimenti.py."""
    pass


class VecFieldWithSteam2(_esperimenti.VecFieldWithSteam2):
    """Prova storica; modificare scene/esperimenti.py."""
    pass


class GreenVisual(_esperimenti.GreenVisual):
    """Prova storica; modificare scene/esperimenti.py."""
    pass


class Green_approx(_esperimenti.Green_approx):
    """Prova storica; modificare scene/esperimenti.py."""
    pass


class ParialApprox(_esperimenti.ParialApprox):
    """Prova storica; modificare scene/esperimenti.py."""
    pass


class Finale(_esperimenti.Finale):
    """Prova storica; modificare scene/esperimenti.py."""
    pass

