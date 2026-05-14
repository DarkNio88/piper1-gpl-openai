#!/bin/bash
# (Debian/Ubuntu) installa tool di build se necessario
apt update
apt install -y build-essential cmake ninja-build

# crea virtualenv e attivalo (consigliato)
#python3 -m venv .venv
#source .venv/bin/activate

# installa builder e dipendenze e compila/installa in editable
python3 -m pip install -U pip setuptools wheel scikit-build
python3 -m pip install -e .[dev,http]
# oppure prova la compilazione inplace
python3 setup.py build_ext --inplace

# avvia il server (senza PYTHONPATH=src)
python3 -m piper.http_server -m it_IT-paola-medium --host 0.0.0.0 --debug