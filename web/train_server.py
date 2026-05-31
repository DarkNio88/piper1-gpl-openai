#!/usr/bin/env python3
"""
Minimal Flask web UI to start Piper training jobs and stream logs.

Usage:
  python3 web/train_server.py

Install requirements in `web/requirements.txt` first.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
import uuid
from typing import Dict

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
)

app = Flask(__name__, template_folder="templates", static_folder="static")

# In-memory job store. For a production service, replace with persistent storage.
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    # Collect form inputs and construct the training command.
    csv_path = request.form.get("csv_path", "").strip()
    audio_dir = request.form.get("audio_dir", "").strip()
    voice_name = request.form.get("voice_name", "").strip()
    sample_rate = request.form.get("sample_rate", "").strip()
    batch_size = request.form.get("batch_size", "").strip()
    ckpt_path = request.form.get("ckpt_path", "").strip()
    extra_args = request.form.get("extra_args", "").strip()

    cmd = ["python3", "-m", "piper.train", "fit"]
    if voice_name:
        cmd += ["--data.voice_name", voice_name]
    if csv_path:
        cmd += ["--data.csv_path", csv_path]
    if audio_dir:
        cmd += ["--data.audio_dir", audio_dir]
    if sample_rate:
        cmd += ["--model.sample_rate", sample_rate]
    if batch_size:
        cmd += ["--data.batch_size", batch_size]
    if ckpt_path:
        cmd += ["--ckpt_path", ckpt_path]
    if extra_args:
        cmd += shlex.split(extra_args)

    job_id = str(uuid.uuid4())[:8]
    job = _make_job_record()
    job["id"] = job_id
    job["cmd"] = " ".join(shlex.quote(c) for c in cmd)
    job["status"] = "running"
    job["start_time"] = time.time()

    # Start the process in the repository root so relative paths work.
    cwd = os.getcwd()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=cwd
    )
    job["proc"] = proc

    with JOBS_LOCK:
        JOBS[job_id] = job

    t = threading.Thread(target=_reader_thread, args=(proc, job_id), daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "cmd": job["cmd"]}), 201


@app.route("/jobs", methods=["GET"])
def list_jobs():
    with JOBS_LOCK:
        return jsonify(
            {
                jid: {
                    "status": j["status"],
                    "cmd": j["cmd"],
                    "started": j["start_time"],
                }
                for jid, j in JOBS.items()
            }
        )


@app.route("/logs/<job_id>")
def logs(job_id: str):
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


@app.route("/stop/<job_id>", methods=["POST"])
def stop(job_id: str):
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Piper training web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=True)
