# Web UI per training Piper

Semplice interfaccia Flask per avviare `python -m piper.train fit` e visualizzare i log in tempo reale.

Installazione rapida:

```sh
python3 -m venv .venv-web
source .venv-web/bin/activate
python -m pip install -r web/requirements.txt
```

Avvio:

```sh
python3 web/train_server.py --host 0.0.0.0 --port 8080
```

Quindi aprire `http://localhost:8080` e compilare i percorsi richiesti (CSV, audio_dir, ecc.).

Nota: il server esegue i processi di training nello stesso ambiente utente; assicurarsi che le dipendenze di training (PyTorch, extras) siano installate nel sistema/venv corretto.
