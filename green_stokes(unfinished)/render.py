#!/usr/bin/env python3
"""Renderizza scene singole, nuove o tutte; non monta i video."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
from scene.catalogo import SCENES, EXPERIMENTS

ROOT = Path(__file__).resolve().parent

def python_for_manim():
    if importlib.util.find_spec('manim'):
        return sys.executable
    candidates = [ROOT/'.venv/bin/python', ROOT/'venv/bin/python',
                  ROOT.parents[1]/'Manim/venv/bin/python']
    for candidate in candidates:
        if candidate.exists():
            probe = subprocess.run([str(candidate),'-c','import manim'],capture_output=True)
            if probe.returncode == 0:
                return str(candidate)
    raise SystemExit('Ambiente Manim non trovato. Attiva il tuo ambiente e riprova.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenes',nargs='*',help='Nomi delle scene, come in main.py')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--new',action='store_true',help='Le sei scene nuove')
    group.add_argument('--all',action='store_true',help='Le dodici scene principali, in ordine')
    parser.add_argument('--preview',action='store_true',help='480p15 in video/480p15, senza toccare i finali')
    parser.add_argument('--open',action='store_true',help='Apri i video al termine del rendering')
    parser.add_argument('--list',action='store_true',help='Elenca scene e comandi')
    args=parser.parse_args()
    if args.scenes and (args.new or args.all):
        parser.error('Scegli nomi di scene oppure --new/--all.')
    if args.list or not (args.scenes or args.new or args.all):
        for i,s in enumerate(SCENES,1):
            print(f'{i:02d}. {s["name"]:28} {s["title"]}'+(' [nuova]' if s['new'] else ''))
        print('\nProve storiche: '+', '.join(EXPERIMENTS))
        print('\nEsempi:\n  python3 render.py StokesSurface --preview --open\n  python3 render.py --new\n  python3 render.py --all')
        return
    selected=([s['name'] for s in SCENES if not args.new or s['new']]
              if args.new or args.all else args.scenes)
    allowed={s['name'] for s in SCENES}|set(EXPERIMENTS)
    unknown=set(selected)-allowed
    if unknown:parser.error('Scene non riconosciute: '+', '.join(sorted(unknown)))
    command=[python_for_manim(),'-m','manim','-ql' if args.preview else '-qh']
    if args.open:command.append('-p')
    command += ['main.py',*selected]
    subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__':
    main()
