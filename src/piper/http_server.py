"""Flask web server with HTTP API for Piper."""

import argparse
import io
import json
import logging
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen
import os

from flask import Flask, request, Response, jsonify, render_template
import subprocess
import threading
import time
import uuid
import shlex

from . import PiperVoice, SynthesisConfig
from .download_voices import VOICES_JSON, download_voice

_LOGGER = logging.getLogger()


def main() -> None:
    """Run HTTP server."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="HTTP server host")
    parser.add_argument("--port", type=int, default=5000, help="HTTP server port")
    #
    parser.add_argument("-m", "--model", required=True, help="Path to Onnx model file")
    #
    parser.add_argument("-s", "--speaker", type=int, help="Id of speaker (default: 0)")
    parser.add_argument(
        "--length-scale", "--length_scale", type=float, help="Phoneme length"
    )
    parser.add_argument(
        "--noise-scale", "--noise_scale", type=float, help="Generator noise"
    )
    parser.add_argument(
        "--noise-w-scale",
        "--noise_w_scale",
        "--noise-w",
        "--noise_w",
        type=float,
        help="Phoneme width noise",
    )
    #
    parser.add_argument("--cuda", action="store_true", help="Use GPU")
    #
    parser.add_argument(
        "--sentence-silence",
        "--sentence_silence",
        type=float,
        default=0.0,
        help="Seconds of silence after each sentence",
    )
    #
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        action="append",
        default=[str(Path.cwd())],
        help="Data directory to check for downloaded models (default: current directory)",
    )
    parser.add_argument(
        "--download-dir",
        "--download_dir",
        help="Path to download voices (default: first data dir)",
    )
    #
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    parser.add_argument(
        "--train-password",
        help="Password required to start/stop training (or set PIPER_TRAIN_PASSWORD)",
    )
    args = parser.parse_args()
    # Training password (optional)
    train_password = args.train_password or os.environ.get("PIPER_TRAIN_PASSWORD")
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    if not args.download_dir:
        # Download voices to first data directory if not specified
        args.download_dir = args.data_dir[0]

    download_dir = Path(args.download_dir)

    # Download voice if file doesn't exist
    model_path = Path(args.model)
    if not model_path.exists():
        # Look in data directories
        voice_name = args.model
        for data_dir in args.data_dir:
            maybe_model_path = Path(data_dir) / f"{voice_name}.onnx"
            _LOGGER.debug("Checking '%s'", maybe_model_path)
            if maybe_model_path.exists():
                model_path = maybe_model_path
                break

    if not model_path.exists():
        raise ValueError(
            f"Unable to find voice: {model_path} (use piper.download_voices)"
        )

    default_model_id = model_path.name.rstrip(".onnx")

    # Load voice
    default_voice = PiperVoice.load(model_path, use_cuda=args.cuda)
    loaded_voices: Dict[str, PiperVoice] = {default_model_id: default_voice}

    def synthesize_bytes(data: Dict[str, Any]) -> bytes:
        """Synthesize audio bytes from a request-like dict.

        Accepts both Piper-style requests (text) and OpenAI-style (input/model).
        """
        # Accept either OpenAI's `input` or Piper's `text`
        text = (data.get("text") or data.get("input") or "").strip()
        if not text:
            raise ValueError("No text provided")

        _LOGGER.debug(data)

        # Voice/model selection: accept `voice` or `model` (OpenAI uses `model`)
        model_id = data.get("voice") or data.get("model") or default_model_id
        voice = loaded_voices.get(model_id)
        if voice is None:
            for data_dir in args.data_dir:
                maybe_model_path = Path(data_dir) / f"{model_id}.onnx"
                if maybe_model_path.exists():
                    _LOGGER.debug("Loading voice %s", model_id)
                    voice = PiperVoice.load(maybe_model_path, use_cuda=args.cuda)
                    loaded_voices[model_id] = voice
                    break

        if voice is None:
            _LOGGER.warning("Voice not found: %s. Using default voice.", model_id)
            voice = default_voice

        speaker_id: Optional[int] = data.get("speaker_id")
        if (voice.config.num_speakers > 1) and (speaker_id is None):
            speaker = data.get("speaker")
            if speaker:
                speaker_id = voice.config.speaker_id_map.get(speaker)

            if speaker_id is None:
                _LOGGER.warning(
                    "Speaker not found: '%s' in %s",
                    speaker,
                    voice.config.speaker_id_map.keys(),
                )
                speaker_id = args.speaker or 0

        if (speaker_id is not None) and (speaker_id > voice.config.num_speakers):
            speaker_id = 0

        syn_config = SynthesisConfig(
            speaker_id=speaker_id,
            length_scale=float(
                data.get(
                    "length_scale",
                    (
                        args.length_scale
                        if args.length_scale is not None
                        else voice.config.length_scale
                    ),
                )
            ),
            noise_scale=float(
                data.get(
                    "noise_scale",
                    (
                        args.noise_scale
                        if args.noise_scale is not None
                        else voice.config.noise_scale
                    ),
                )
            ),
            noise_w_scale=float(
                data.get(
                    "noise_w_scale",
                    (
                        args.noise_w_scale
                        if args.noise_w_scale is not None
                        else voice.config.noise_w_scale
                    ),
                )
            ),
        )

        _LOGGER.debug("Synthesizing text: '%s' with config=%s", text, syn_config)
        with io.BytesIO() as wav_io:
            wav_file: Optional[wave.Wave_write] = None
            wav_params_set = False
            try:
                for i, audio_chunk in enumerate(voice.synthesize(text, syn_config)):
                    if wav_file is None:
                        # Open WAV writer lazily once we have the first chunk
                        wav_file = wave.open(wav_io, "wb")
                        wav_file.setframerate(audio_chunk.sample_rate)
                        wav_file.setsampwidth(audio_chunk.sample_width)
                        wav_file.setnchannels(audio_chunk.sample_channels)
                        wav_params_set = True

                    if i > 0:
                        wav_file.writeframes(
                            bytes(
                                int(
                                    voice.config.sample_rate * args.sentence_silence * 2
                                )
                            )
                        )

                    wav_file.writeframes(audio_chunk.audio_int16_bytes)

            finally:
                if wav_file is not None:
                    wav_file.close()

            if not wav_params_set:
                # No audio produced (synthesis failed early)
                raise ValueError("No audio produced by voice.synthesize()")

            return wav_io.getvalue()

    # Create web server (serve web/ templates and static for UI)
    repo_root = Path(__file__).resolve().parents[2]
    template_folder = str(repo_root / "web" / "templates")
    static_folder = str(repo_root / "web" / "static")
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)

    # Reload templates automatically when they change and avoid static caching
    # so HTML/CSS edits are reflected without restarting the server.
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True
    # Reduce static file caching (useful during development)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    # Protect training endpoints if a training password is configured.
    def _train_auth_ok() -> bool:
        if not train_password:
            return True

        # Check Authorization: Bearer <password>
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            if token == train_password:
                return True

        # Check HTTP Basic auth (password match)
        auth = request.authorization
        if auth and auth.password == train_password:
            return True

        # Check JSON or form field 'train_password'
        # Use silent=True to avoid raising on non-JSON requests
        try:
            body = request.get_json(silent=True) or {}
            if body.get("train_password") == train_password:
                return True
        except Exception:
            pass

        if request.form and request.form.get("train_password") == train_password:
            return True

        if request.args.get("train_password") == train_password:
            return True

        return False

    @app.before_request
    def _require_train_auth():
        # Only enforce for training endpoints
        if request.path.startswith("/train"):
            if _train_auth_ok():
                return None
            return Response("Unauthorized", 401, {"WWW-Authenticate": 'Basic realm="Piper Training"'})

    @app.route("/", methods=["GET"])
    def app_index():
        """Serve training UI HTML."""
        return render_template("index.html")

    # Training job management (start/list/logs/stop)
    JOBS: Dict[str, dict] = {}
    JOBS_LOCK = threading.Lock()


    def _make_job_record() -> dict:
        return {
            "id": None,
            "cmd": None,
            "proc": None,
            "logs": [],
            "status": "queued",
            "start_time": None,
            "end_time": None,
        }


    def _reader_thread(proc: subprocess.Popen, job_id: str) -> None:
        # Stream stdout lines into the job logs.
        try:
            with proc.stdout:
                for line in iter(proc.stdout.readline, ""):
                    if not line:
                        break
                    with JOBS_LOCK:
                        JOBS[job_id]["logs"].append(line.rstrip())
        finally:
            ret = proc.wait()
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "finished" if ret == 0 else "error"
                JOBS[job_id]["end_time"] = time.time()

    @app.route("/voices", methods=["GET"])
    def app_voices() -> Dict[str, Any]:
        """List downloaded voices.

        Outputs a JSON object with the format:
        {
          "<voice name>": { <voice config> },
          ...
        }

        for each voice in your data directories.
        """
        voices_dict: Dict[str, Any] = {}
        config_paths: List[Path] = [Path(f"{model_path}.json")]

        for data_dir in args.data_dir:
            for onnx_path in Path(data_dir).glob("*.onnx"):
                config_path = Path(f"{onnx_path}.json")
                if config_path.exists():
                    config_paths.append(config_path)

        for config_path in config_paths:
            model_id = config_path.name.rstrip(".onnx.json")
            if model_id in voices_dict:
                continue

            with open(config_path, "r", encoding="utf-8") as config_file:
                voices_dict[model_id] = json.load(config_file)

        return voices_dict

    @app.route("/all-voices", methods=["GET"])
    def app_all_voices() -> Dict[str, Any]:
        """List all Piper voices.

        Outputs voices.json from the piper-voices repo on HuggingFace.
        See: https://huggingface.co/rhasspy/piper-voices
        """
        with urlopen(VOICES_JSON) as response:
            return json.load(response)

    @app.route("/download", methods=["POST"])
    def app_download() -> str:
        """Download a voice.

        Downloads the .onnx and .onnx.json file from piper-voices repo on HuggingFace.
        See: https://huggingface.co/rhasspy/piper-voices

        Expects a JSON object with the format:
        {
          "voice": "<voice name>",   (required)
          "force_redownload": false  (optional)
        }

        Returns the name of the voice.
        Voice format must be <language>-<name>-<quality> like "en_US-lessac-medium".
        """
        data = json.loads(request.data)
        model_id = data.get("voice")
        if not model_id:
            raise ValueError("voice is required")

        force_redownload = data.get("force_redownload", False)
        download_voice(model_id, download_dir, force_redownload=force_redownload)

        return model_id

    @app.route("/", methods=["POST"])
    def app_synthesize() -> bytes:
        """Synthesize audio from text.

        Expects a JSON object with the format:
        {
          "text": "Text to speak.",      (required)
          "voice": "<voice name>",       (optional)
          "speaker": "<speaker name>",   (optional)
          "speaker_id": "<speaker id>",  (optional, overrides speaker)
          "length_scale": 1.0,           (optional)
          "noise_scale": 0.667,          (optional)
          "length_w_scale": 0.8          (optional)
        }
        """
        try:
            data = json.loads(request.data)
            wav = synthesize_bytes(data)
            return app.response_class(wav, mimetype="audio/wav")
        except Exception:
            _LOGGER.exception("Error in app_synthesize")
            import traceback

            tb = traceback.format_exc()
            return app.response_class(tb, mimetype="text/plain", status=500)

    @app.route("/v1/audio/speech", methods=["POST"])
    def app_openai_speech() -> bytes:
        """OpenAI-compatible TTS endpoint.

        Accepts JSON like {"input": "text", "model": "voice-name", ...}
        and returns `audio/wav` bytes.
        """
        try:
            data = request.get_json(force=True)
            wav = synthesize_bytes(data)
            return app.response_class(wav, mimetype="audio/wav")
        except Exception:
            _LOGGER.exception("Error in app_openai_speech")
            import traceback

            tb = traceback.format_exc()
            return app.response_class(tb, mimetype="text/plain", status=500)

    # ------------------------
    # Training control API
    # ------------------------
    @app.route("/train/start", methods=["POST"])
    def app_train_start():
        # Accept JSON body for training parameters.
        data = request.get_json(force=True) or {}

        voice_name = data.get("voice_name") or data.get("voice")
        csv_path = data.get("csv_path")
        audio_dir = data.get("audio_dir")
        sample_rate = data.get("sample_rate")
        batch_size = data.get("batch_size")
        ckpt_path = data.get("ckpt_path")
        extra_args = data.get("extra_args", "")

        cmd = ["python3", "-m", "piper.train", "fit"]
        if voice_name:
            cmd += ["--data.voice_name", str(voice_name)]
        if csv_path:
            cmd += ["--data.csv_path", str(csv_path)]
        if audio_dir:
            cmd += ["--data.audio_dir", str(audio_dir)]
        if sample_rate:
            cmd += ["--model.sample_rate", str(sample_rate)]
        if batch_size:
            cmd += ["--data.batch_size", str(batch_size)]
        if ckpt_path:
            cmd += ["--ckpt_path", str(ckpt_path)]
        if extra_args:
            cmd += shlex.split(str(extra_args))

        job_id = str(uuid.uuid4())[:8]
        job = _make_job_record()
        job["id"] = job_id
        job["cmd"] = " ".join(shlex.quote(c) for c in cmd)
        job["status"] = "running"
        job["start_time"] = time.time()

        # Start the process in the repository root so relative paths work.
        cwd = Path.cwd()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(cwd)
        )
        job["proc"] = proc

        with JOBS_LOCK:
            JOBS[job_id] = job

        t = threading.Thread(target=_reader_thread, args=(proc, job_id), daemon=True)
        t.start()

        return jsonify({"job_id": job_id, "cmd": job["cmd"]}), 201


    @app.route("/train/jobs", methods=["GET"])
    def app_train_jobs():
        with JOBS_LOCK:
            return jsonify(
                {
                    jid: {"status": j["status"], "cmd": j["cmd"], "started": j["start_time"]}
                    for jid, j in JOBS.items()
                }
            )


    @app.route("/train/logs/<job_id>")
    def app_train_logs(job_id: str):
        # Server-Sent Events stream of log lines for the job.
        def generate():
            last = 0
            while True:
                with JOBS_LOCK:
                    j = JOBS.get(job_id)
                    if not j:
                        yield "data: [ERROR] job not found\n\n"
                        return
                    logs = list(j.get("logs", []))
                    status = j.get("status", "unknown")
                if last < len(logs):
                    for line in logs[last:]:
                        yield f"data: {line}\n\n"

                    last = len(logs)
                if status in ("finished", "error", "cancelled"):
                    yield f"data: [END] status={status}\n\n"

                    break
                time.sleep(0.5)

        return Response(generate(), mimetype="text/event-stream")


    @app.route("/train/stop/<job_id>", methods=["POST"])
    def app_train_stop(job_id: str):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return jsonify({"error": "job not found"}), 404
            proc = job.get("proc")
            if proc and proc.poll() is None:
                proc.terminate()
                job["status"] = "cancelled"
                return jsonify({"status": "terminated"})
            return jsonify({"status": "not running"})

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
