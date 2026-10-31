"""
launcher.py
-----------
Entry point for the packaged ECE-LASU-LASW .exe.

What it does:
  1. Sets OLLAMA_MODELS env var to the bundled models folder
  2. Starts the bundled Ollama server in the background
  3. Waits for Ollama to be ready (polls /api/tags)
  4. Opens the browser to http://localhost:5000
  5. Starts the Flask app (blocking)

On window close / Ctrl-C, it kills the Ollama process cleanly.
"""

import os
import sys
import time
import signal
import subprocess
import webview
import threading
import urllib.request
import urllib.error
import multiprocessing
multiprocessing.freeze_support()  # ← Must be first thing in frozen exe

# ── Resolve base directory ──────────────────────────────────────────────
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Paths ───────────────────────────────────────────────────────────────
OLLAMA_EXE    = os.path.join(BASE_DIR, "ollama", "ollama.exe")
OLLAMA_MODELS = os.path.join(BASE_DIR, "ollama", "models")
FLASK_PORT    = 5000
OLLAMA_PORT   = 11434

# ── Point Ollama at the bundled models folder ───────────────────────────
os.environ["OLLAMA_MODELS"] = OLLAMA_MODELS
os.environ["OLLAMA_HOST"]   = f"127.0.0.1:{OLLAMA_PORT}"

ollama_proc = None


def start_ollama():
    """Launch ollama serve as a background process."""
    global ollama_proc
    if not os.path.exists(OLLAMA_EXE):
        print(f"[WARN] Ollama binary not found at: {OLLAMA_EXE}")
        print("       LLM features will be unavailable.")
        return

    # Kill any orphaned ollama.exe from a previous ungraceful exit,
    # so we never stack multiple server instances on relaunch.
    subprocess.run(
        ["taskkill", "/F", "/IM", "ollama.exe", "/T"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    print("[INFO] Starting Ollama server...")
    ollama_proc = subprocess.Popen(
        [OLLAMA_EXE, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW
    )


def wait_for_ollama(timeout: int = 30) -> bool:
    url = f"http://127.0.0.1:{OLLAMA_PORT}/api/tags"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            print("[INFO] Ollama is ready.")
            return True
        except:
            time.sleep(2)  # ← was 1, now 2 — less aggressive polling
    print("[WARN] Ollama not ready yet — will retry in background.")
    # Keep retrying silently in background
    def retry():
        for _ in range(30):
            time.sleep(5)
            try:
                urllib.request.urlopen(url, timeout=2)
                print("[INFO] Ollama became ready (delayed).")
                return
            except:
                continue
    threading.Thread(target=retry, daemon=True).start()
    return False


def shutdown(signum=None, frame=None):
    """Clean up Ollama on exit."""
    global ollama_proc
    if ollama_proc and ollama_proc.poll() is None:
        print("[INFO] Shutting down Ollama...")
        ollama_proc.terminate()
        try:
            ollama_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ollama_proc.kill()
    sys.exit(0)


# Register clean-up handlers
signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

def main():
    # 1. Start Ollama
    start_ollama()
    wait_for_ollama(timeout=30)

    # 2. Start Flask in a background thread
    import threading
    from app import app, initialize_app_services, create_local_index, BASE_DIR as APP_BASE

    # Create FAISS index if missing
    index_dir = os.path.join(APP_BASE, "radiology_faiss_index")
    if not os.path.exists(index_dir):
        pdf_dir = os.path.join(APP_BASE, "pdf")
        pdf_paths = []
        if os.path.isdir(pdf_dir):
            pdf_paths = [
                os.path.join(pdf_dir, f)
                for f in os.listdir(pdf_dir)
                if f.lower().endswith(".pdf")
            ]
        if pdf_paths:
            create_local_index(pdf_paths)
            
    initialize_app_services()

    from app import APP_READY_STATE
    APP_READY_STATE["server_ready"] = True

    flask_thread = threading.Thread(
        target=lambda: app.run(
            debug=False,
            host="0.0.0.0",
            port=5000,
            threaded=True,
            use_reloader=False
        ),
        daemon=True
    )
    flask_thread.start()

    # 3. Wait for Flask to be ready
    import urllib.request, time
    for _ in range(20):
        try:
            urllib.request.urlopen("http://localhost:5000", timeout=2)
            break
        except:
            time.sleep(1)

    # 4. Open native desktop window instead of browser
    window = webview.create_window(
        title="ECE-LASU-LASW",
        url="http://localhost:5000",
        width=1280,
        height=800,
        resizable=True,
        min_size=(900, 600),
    )
    webview.start()  # This blocks until window is closed

    # 5. Clean up when window closes
    shutdown()

if __name__ == "__main__":
    main()