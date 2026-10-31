import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import os
import re
import uuid
import time
import json
import hashlib
import subprocess
import requests as http_requests   # aliased — you already use `from flask import request`
import nibabel as nib
import numpy as np
import zipfile
import torch
import operator
import traceback
import base64
import threading
import builtins
from collections import deque
from flask import send_from_directory, send_file
from werkzeug.utils import secure_filename
from typing import TypedDict, Annotated, List
from flask import Flask, request, jsonify
from flask_cors import CORS
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

# ────────────────────────────────────────────────────────────────────────
# LOG CAPTURE — feeds the in-app console drawer (replaces visible terminal)
# Every print() call is captured here AND still goes to the real terminal
# (or to PyInstaller's hidden console buffer when console=False).
# ────────────────────────────────────────────────────────────────────────
LOG_BUFFER = deque(maxlen=500)
_original_print = builtins.print

def _captured_print(*args, **kwargs):
    message = " ".join(str(a) for a in args)
    LOG_BUFFER.append({"time": time.strftime("%H:%M:%S"), "text": message})
    _original_print(*args, **kwargs)

builtins.print = _captured_print

# Tracks whether all background services have finished loading —
# used by the frontend loading screen via /api/ready
APP_READY_STATE = {
    "rag_ready": False,
    "sdxl_ready": False,
    "sdxl_attempted": False,
    "server_ready": False
}

# LangChain / Ollama Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS

# Chain Imports
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AnyMessage

# LangGraph Imports
from langgraph.graph import StateGraph, END

# ────────────────────────────────────────────────────────────────────────
# BASE DIRECTORY — works both as a .py script and as a PyInstaller .exe
# ────────────────────────────────────────────────────────────────────────
if getattr(sys, 'frozen', False):
    # Running as compiled .exe — use the folder containing the .exe
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # Running as a normal Python script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ────────────────────────────────────────────────────────────────────────
# App Config
# ────────────────────────────────────────────────────────────────────────
app = Flask(__name__,
            static_folder=BASE_DIR,
            template_folder=BASE_DIR)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Globals
RAG_CHAIN_OBJ = None
AGENT_WORKFLOW = None
LOCAL_LLM = None
img_pipe = None

# ────────────────────────────────────────────────────────────────────────
# File Upload / Output Folders — relative to BASE_DIR
# ────────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# ────────────────────────────────────────────────────────────────────────
# CT Scan Folders + Job Tracking
# ────────────────────────────────────────────────────────────────────────
CT_UPLOAD_FOLDER = os.path.join(BASE_DIR, 'ct_uploads')
CT_OUTPUT_FOLDER = os.path.join(BASE_DIR, 'ct_output')
os.makedirs(CT_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CT_OUTPUT_FOLDER, exist_ok=True)

# In-memory job tracker — polled by the frontend via /api/ct_status/<job_id>
# stage values: uploading | freeing_memory | segmenting | generating_report |
#               reloading_sdxl | done | error
CT_JOBS = {}

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    relative_path = f"uploads/{unique_filename}"
    return jsonify({
        'message': 'File successfully uploaded',
        'filepath': filepath,
        'relative_path': relative_path
    }), 200

@app.route('/api/upload_ct', methods=['POST'])
def upload_ct_series():
    """Accepts either a full DICOM series (multiple loose files from a folder
    picker) or a single .zip archive containing the series — saves everything
    into its own subfolder under ct_uploads/."""
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files received'}), 400

    scan_id = uuid.uuid4().hex[:8]
    scan_folder = os.path.join(CT_UPLOAD_FOLDER, f"scan_{scan_id}")
    os.makedirs(scan_folder, exist_ok=True)

    saved = 0
    for f in files:
        if not f.filename:
            continue

        if f.filename.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(f.stream, 'r') as zf:
                    for member in zf.namelist():
                        if member.endswith('/') or '__MACOSX' in member:
                            continue
                        safe_name = secure_filename(os.path.basename(member))
                        if not safe_name:
                            continue
                        with zf.open(member) as src, open(os.path.join(scan_folder, safe_name), 'wb') as dst:
                            dst.write(src.read())
                        saved += 1
            except zipfile.BadZipFile:
                return jsonify({'error': f'{f.filename} is not a valid zip file'}), 400
            continue

        safe_name = secure_filename(os.path.basename(f.filename))
        if not safe_name:
            continue
        f.save(os.path.join(scan_folder, safe_name))
        saved += 1

    if saved == 0:
        return jsonify({'error': 'No valid files saved'}), 400

    print(f"[INFO] CT series uploaded: {saved} file(s) -> {scan_folder}")
    return jsonify({'message': 'CT series uploaded', 'dicom_folder': scan_folder, 'file_count': saved}), 200

@app.route('/uploads/<filename>')
def serve_uploaded_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/output/<filename>')
def serve_generated_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

# ────────────────────────────────────────────────────────────────────────
# Console & Readiness Endpoints — power the in-app console drawer and
# the loading screen, replacing the visible terminal window
# ────────────────────────────────────────────────────────────────────────
@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Returns recent log lines for the in-app console drawer."""
    return jsonify({"logs": list(LOG_BUFFER)})

@app.route('/api/ready', methods=['GET'])
def get_ready_state():
    """Returns service readiness flags for the loading screen."""
    return jsonify(APP_READY_STATE)

# ────────────────────────────────────────────────────────────────────────
# Persistent Chat History — saved to disk so it survives app restarts.
# PyWebView's localStorage is wiped on close; a JSON file is not.
# ────────────────────────────────────────────────────────────────────────
HISTORY_FILE = os.path.join(BASE_DIR, 'chat_history.json')

def load_history_from_disk():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read history file: {e}")
    return {}

def save_history_to_disk(data):
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save history: {e}")

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify(load_history_from_disk())

@app.route('/api/history', methods=['POST'])
def post_history():
    data = request.get_json(silent=True) or {}
    save_history_to_disk(data)
    return jsonify({'status': 'saved'})

# ────────────────────────────────────────────────────────────────────────
# Tool 1: Image Generation (SDXL Turbo)
# ────────────────────────────────────────────────────────────────────────
import difflib

CACHE_MANIFEST_PATH = os.path.join(OUTPUT_FOLDER, "image_cache_manifest.json")

def _load_cache_manifest():
    if os.path.exists(CACHE_MANIFEST_PATH):
        try:
            with open(CACHE_MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_cache_manifest(manifest):
    try:
        with open(CACHE_MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save image cache manifest: {e}")

def _find_similar_cached_prompt(normalized_prompt: str, manifest: dict, threshold: float = 0.87):
    best_key, best_score = None, 0.0
    for key, entry in manifest.items():
        score = difflib.SequenceMatcher(None, normalized_prompt, entry["normalized_prompt"]).ratio()
        if score > best_score:
            best_score, best_key = score, key
    if best_key and best_score >= threshold:
        return best_key, manifest[best_key]["filename"], best_score
    return None

@tool
def generate_image(prompt: str) -> str:
    """Generate an image from a detailed text prompt using Stable Diffusion XL Turbo.
    Automatically returns a cached image if a similar prompt was used before,
    making repeated requests instant with no regeneration needed."""
    global img_pipe

    normalized = re.sub(r'\s+', ' ', prompt.strip().lower())
    cache_key = hashlib.md5(normalized.encode('utf-8')).hexdigest()
    cache_filename = f"cache_{cache_key}.png"
    cache_path = os.path.join(OUTPUT_FOLDER, cache_filename)
    manifest = _load_cache_manifest()

    # 1. Exact hash match (fast path — your original logic)
    if os.path.exists(cache_path):
        print(f"[INFO] Exact cache hit for prompt: '{prompt[:50]}...'")
        return f"/output/{cache_filename}"

    # 2. Fuzzy match against previously cached prompts (handles LLM rewording)
    match = _find_similar_cached_prompt(normalized, manifest)
    if match:
        matched_key, matched_filename, score = match
        matched_path = os.path.join(OUTPUT_FOLDER, matched_filename)
        if os.path.exists(matched_path):
            print(f"[INFO] Fuzzy cache hit ({score:.2f}) → reusing {matched_filename}")
            return f"/output/{matched_filename}"

    # 3. True cache miss — generate
    if img_pipe is None:
        return "Error: Image generation model is not loaded yet. Please wait and try again."

    try:
        print(f"[INFO] Generating new image for prompt: '{prompt[:50]}...'")
        with torch.inference_mode():
            image = img_pipe(
                prompt=prompt,
                num_inference_steps=1,
                guidance_scale=0.0,
                height=512,
                width=512,
            ).images[0]

        image.save(cache_path)
        manifest[cache_key] = {"normalized_prompt": normalized, "filename": cache_filename}
        _save_cache_manifest(manifest)
        print(f"[INFO] Image saved to cache: {cache_filename}")
        return f"/output/{cache_filename}"

    except Exception as e:
        print(f"[ERROR] Image generation failed: {e}")
        return f"Image generation failed: {str(e)}"
    

# ────────────────────────────────────────────────────────────────────────
# Tool 2: Vision Analysis (Moondream)
# ────────────────────────────────────────────────────────────────────────
@tool
def analyze_radiology_image(image_path: str, query: str = "Describe this medical image in detail") -> str:
    """Analyzes a local image file using Moondream Vision LLM."""
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        llm = ChatOllama(model="moondream")
        message = HumanMessage(content=[
            {"type": "text", "text": query},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ])
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        return f"Vision analysis failed: {str(e)}"

# ────────────────────────────────────────────────────────────────────────
# Tool 3: RAG Search
# ────────────────────────────────────────────────────────────────────────
@tool
def search_radiology_knowledge(query: str) -> str:
    """Search the radiology knowledge base for relevant information about diseases,
    conditions, imaging findings, diagnoses, and treatment protocols."""
    global RAG_CHAIN_OBJ
    if RAG_CHAIN_OBJ is None:
        return "Knowledge base not available."
    try:
        result = RAG_CHAIN_OBJ.invoke({"input": query})
        return result.get("answer", "No answer found.")
    except Exception as e:
        return f"Search failed: {str(e)}"

# ────────────────────────────────────────────────────────────────────────
# Tool 4: 3D CT Volumetric Analysis (TotalSegmentator, isolated subprocess)
# ────────────────────────────────────────────────────────────────────────
OLLAMA_MODELS = ["qwen2.5:3b", "moondream", "nomic-embed-text"]

def unload_ollama_models():
    """Force Ollama to drop models from memory immediately (keep_alive=0)."""
    for model in OLLAMA_MODELS:
        try:
            http_requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=5
            )
            print(f"[INFO] Unloaded Ollama model: {model}")
        except Exception as e:
            print(f"[WARN] Could not unload {model}: {e}")

def unload_sdxl():
    """Drop SDXL from memory. Runs in-process — its reload path already
    exists and is cheap to re-trigger via load_sdxl_background()."""
    global img_pipe
    if img_pipe is not None:
        del img_pipe
        img_pipe = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[INFO] SDXL unloaded")

def run_ct_segmentation(dicom_folder: str, job_id: str) -> str:
    output_dir = os.path.join(CT_OUTPUT_FOLDER, f"seg_{job_id}")
    os.makedirs(output_dir, exist_ok=True)

    if getattr(sys, 'frozen', False):
        cmd = [os.path.join(BASE_DIR, "TotalSegRunner", "TotalSegRunner.exe"), dicom_folder, output_dir]
    else:
        script_path = os.path.join(BASE_DIR, "run_totalsegmentator.py")
        cmd = [sys.executable, script_path, dicom_folder, output_dir]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW
    )

    output_lines = []
    for line in process.stdout:
        line = line.rstrip()
        if line:
            print(f"[TotalSegRunner] {line}")  # streams into your console drawer live
            output_lines.append(line)

    process.wait(timeout=1800)
    full_output = "\n".join(output_lines)

    if "RESULT_OK" not in full_output:
        raise RuntimeError(f"Segmentation failed: {full_output}")

    return output_dir

def summarize_segmentation(seg_output_dir: str) -> dict:
    """Reads TotalSegmentator's per-organ NIfTI masks and computes volumes.
    Returns {"structures": {organ_name: volume_ml, ...}, "detected_count": n}."""
    findings = {"structures": {}, "detected_count": 0}

    if not os.path.isdir(seg_output_dir):
        return findings

    for filename in sorted(os.listdir(seg_output_dir)):
        if not filename.endswith((".nii", ".nii.gz")):
            continue
        organ_name = filename.replace(".nii.gz", "").replace(".nii", "").replace("_", " ")
        try:
            img = nib.load(os.path.join(seg_output_dir, filename))
            data = img.get_fdata()
            voxel_count = int(np.count_nonzero(data))
            if voxel_count == 0:
                continue  # structure not present/detected in this scan
            voxel_volume_mm3 = float(np.prod(img.header.get_zooms()))
            volume_ml = round((voxel_count * voxel_volume_mm3) / 1000.0, 1)
            findings["structures"][organ_name] = volume_ml
            findings["detected_count"] += 1
        except Exception as e:
            print(f"[WARN] Could not read {filename}: {e}")

    return findings

def build_ct_report_prompt(findings: dict, patient_info: dict) -> str:
    """Turns raw segmentation numbers into a prompt for the LLM to write
    the actual narrative report."""
    patient_line = ""
    if patient_info:
        name = patient_info.get("name", "")
        sex = patient_info.get("sex", "")
        age = patient_info.get("age", "")
        case_id = patient_info.get("caseId", "")
        patient_line = f"Patient: {name}, {sex}, {age} years old"
        if case_id:
            patient_line += f" (Case ID: {case_id})"

    structures_text = "\n".join(
        f"- {name.title()}: {volume} mL" for name, volume in findings["structures"].items()
    ) or "No structures were confidently segmented."

    return (
        "You are an expert radiologist AI writing a structured report from automated "
        "CT segmentation output. IMPORTANT: this data describes anatomical structures "
        "and their volumes only — it does NOT include pathology detection. Do not "
        "invent findings of disease that aren't supported by the data below.\n\n"
        f"{patient_line}\n\n"
        f"Segmented structures and volumes:\n{structures_text}\n\n"
        "Write a report with these sections: Patient Information, Technique "
        "(note: automated volumetric segmentation, TotalSegmentator fast mode), "
        "Findings (describe the volumes above in clinical language, noting anything "
        "outside typical reference ranges), and Limitations (state clearly that this "
        "is anatomical/volumetric only, not a diagnostic pathology read, and should "
        "be reviewed by a radiologist)."
    )

def process_ct_job(job_id: str, dicom_folder: str, patient_info: dict):
    """Background worker: unload → segment → reload Ollama → report → reload SDXL."""
    try:
        CT_JOBS[job_id] = {"stage": "freeing_memory"}
        unload_ollama_models()
        unload_sdxl()

        CT_JOBS[job_id] = {"stage": "segmenting"}
        seg_output_dir = run_ct_segmentation(dicom_folder, job_id)

        CT_JOBS[job_id] = {"stage": "generating_report"}
        findings = summarize_segmentation(seg_output_dir)
        prompt = build_ct_report_prompt(findings, patient_info)

        llm = ChatOllama(model="qwen2.5:3b")  # loads back into memory automatically
        response = llm.invoke([HumanMessage(content=prompt)])

        CT_JOBS[job_id] = {
            "stage": "done",
            "report": response.content,
            "findings": findings
        }

        # Reload SDXL quietly in the background — user already has their report
        threading.Thread(target=load_sdxl_background, daemon=True).start()

    except subprocess.TimeoutExpired:
        CT_JOBS[job_id] = {"stage": "error", "error": "Segmentation timed out."}
    except Exception as e:
        traceback.print_exc()
        CT_JOBS[job_id] = {"stage": "error", "error": str(e)}

# ────────────────────────────────────────────────────────────────────────
# LangGraph Agent
# ────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

TOOLS = [generate_image, analyze_radiology_image, search_radiology_knowledge]
TOOLS_MAP = {t.name: t for t in TOOLS}

# FIX 3 — Richer system prompt for the RAG chain so context is used fully
def create_rag_chain_object(vectorstore):
    llm = ChatOllama(model="qwen2.5:3b")

    # FIX 4 — Increased k from 3 → 6 so more of the PDF content is retrieved
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 12, "lambda_mult": 0.7}
    )

    system_prompt = (
        "You are an expert radiologist AI assistant with access to authoritative radiology "
        "textbooks and reference materials. Use ONLY the retrieved context below to answer "
        "the question. Be thorough, clinical, and precise. If the context contains relevant "
        "information, always use it — never say you don't know if the answer is present in "
        "the context. Structure your answers clearly with findings, clinical significance, "
        "and management notes where appropriate.\n\nContext:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# FIX 3 — Agent system prompt forces RAG search before answering
AGENT_SYSTEM_PROMPT = SystemMessage(content=(
    "You are an expert radiology AI assistant. You have access to three tools:\n"
    "1. search_radiology_knowledge — searches a knowledge base of authoritative radiology "
    "textbooks. You MUST call this tool first for ANY question about diseases, conditions, "
    "imaging findings, diagnoses, symptoms, treatments, or radiology reports. "
    "Never answer medical questions from memory alone.\n"
    "2. analyze_radiology_image — analyzes an uploaded medical image using vision AI. "
    "Call this when the user uploads a scan or image for analysis.\n"
    "3. generate_image — generates a medical illustration from a text description. "
    "Call this when the user asks to visualize or illustrate a concept.\n\n"
    "Always search the knowledge base first. Then provide a thorough, well-structured answer "
    "based on what you find."
))

def call_model(state: AgentState):
    llm = LOCAL_LLM.bind_tools(TOOLS)
    # FIX 3 — Prepend system prompt to every agent invocation
    messages = [AGENT_SYSTEM_PROMPT] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def call_tool(state: AgentState):
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool_fn = TOOLS_MAP.get(tool_call["name"])
        if tool_fn:
            result = tool_fn.invoke(tool_call["args"])
        else:
            result = f"Unknown tool: {tool_call['name']}"
        results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    return {"messages": results}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "call_tool"
    return END

def setup_langgraph_agent(vectorstore):
    global RAG_CHAIN_OBJ, LOCAL_LLM
    LOCAL_LLM = ChatOllama(model="qwen2.5:3b")
    RAG_CHAIN_OBJ = create_rag_chain_object(vectorstore)
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("call_tool", call_tool)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("call_tool", "agent")
    return workflow.compile()

def load_sdxl_background():
    global img_pipe
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        sdxl_paths = [os.path.join(BASE_DIR, "sdxl_model"), "./sdxl_model"]
        for path in sdxl_paths:
            if os.path.exists(path):
                print(f"[INFO] Loading SDXL from: {path}")
                img_pipe = StableDiffusionXLPipeline.from_pretrained(
                    path, torch_dtype=dtype, use_safetensors=True, local_files_only=True
                )
                img_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                    img_pipe.scheduler.config, timestep_spacing="trailing"
                )
                img_pipe.to(device)
                if device == "cpu":
                    img_pipe.unet = img_pipe.unet.to(memory_format=torch.channels_last)
                print(f"[INFO] SDXL loaded on {device}")
                break
    except Exception as e:
        print(f"[WARN] SDXL failed: {e}")
        img_pipe = None
    finally:
        APP_READY_STATE["sdxl_ready"] = True
        APP_READY_STATE["sdxl_attempted"] = True


# ────────────────────────────────────────────────────────────────────────
# Initialization Services
# ────────────────────────────────────────────────────────────────────────
def initialize_app_services():
    global AGENT_WORKFLOW, img_pipe

    # 1. Load vector store — always relative to BASE_DIR
    INDEX_PATH = os.path.join(BASE_DIR, "radiology_faiss_index")

    if os.path.exists(INDEX_PATH):
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = FAISS.load_local(
                INDEX_PATH, embeddings, allow_dangerous_deserialization=True
            )
            AGENT_WORKFLOW = setup_langgraph_agent(vectorstore)
            print("RAG + Agent initialized")
        except Exception as e:
            print(f"RAG failed: {e}")
            AGENT_WORKFLOW = None
    else:
        print("No FAISS index found - RAG disabled")

    APP_READY_STATE["rag_ready"] = True

    threading.Thread(target=load_sdxl_background, daemon=True).start()
    print("[INFO] SDXL loading in background...")

# ────────────────────────────────────────────────────────────────────────
# Index Creation
# ────────────────────────────────────────────────────────────────────────
def create_local_index(file_paths: list):
    print("Creating document index...")
    all_documents = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        try:
            loader = PyPDFLoader(path)
            all_documents.extend(loader.load())
            print(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            print(f"Failed {path}: {e}")

    if not all_documents:
        print("No documents loaded")
        return False

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(all_documents)
        print(f"{len(chunks)} chunks → Embedding...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        index_save_path = os.path.join(BASE_DIR, "radiology_faiss_index")
        vectorstore.save_local(index_save_path)
        print("Index saved!")
        return True
    except Exception as e:
        print(f"Index failed: {e}")
        traceback.print_exc()
        return False

# ────────────────────────────────────────────────────────────────────────
# Chat Endpoint
# ────────────────────────────────────────────────────────────────────────
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'reply': 'Please provide a message.'}), 400

    if AGENT_WORKFLOW is None:
        try:
            llm = ChatOllama(model="qwen2.5:3b")
            response = llm.invoke([HumanMessage(content=user_message)])
            return jsonify({'reply': response.content, 'image_url': None})
        except:
            return jsonify({'reply': 'Service temporarily unavailable.', 'image_url': None}), 503

    try:
        print(f"User: {user_message[:50]}...")
        result = AGENT_WORKFLOW.invoke({"messages": [HumanMessage(content=user_message)]})
        final_reply = result["messages"][-1].content

        image_url = None
        image_pattern = re.compile(r'(/output/[\w\-]+\.(?:png|jpg|jpeg|gif))', re.IGNORECASE)

        for msg in result["messages"]:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            match = image_pattern.search(content)
            if match:
                image_url = match.group(1)
                print(f"Image URL extracted: {image_url}")
                break

        print(f"Bot: {final_reply[:50]}...")
        return jsonify({'reply': final_reply, 'image_url': image_url})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'reply': f'Sorry, something went wrong: {str(e)}', 'image_url': None}), 500

@app.route('/api/analyze_ct', methods=['POST'])
def analyze_ct():
    data = request.get_json(silent=True) or {}
    dicom_folder = data.get('dicom_folder')
    patient_info = data.get('patient_info', {})

    if not dicom_folder or not os.path.isdir(dicom_folder):
        return jsonify({'error': 'Invalid dicom_folder path'}), 400

    job_id = uuid.uuid4().hex[:8]
    CT_JOBS[job_id] = {"stage": "queued"}

    threading.Thread(
        target=process_ct_job, args=(job_id, dicom_folder, patient_info), daemon=True
    ).start()

    return jsonify({'job_id': job_id}), 202

@app.route('/api/ct_status/<job_id>', methods=['GET'])
def ct_status(job_id):
    job = CT_JOBS.get(job_id)
    if job is None:
        return jsonify({'stage': 'unknown', 'error': 'Job not found'}), 404
    return jsonify(job)

@app.route('/fontawesome/<path:filename>')
def serve_fontawesome(filename):
    fa_dir = os.path.join(BASE_DIR, 'fontawesome')
    return send_from_directory(fa_dir, filename)

# ────────────────────────────────────────────────────────────────────────
# Frontend Routes
# ────────────────────────────────────────────────────────────────────────
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    file_map = {
        '': 'gpt.html',
        'index.html': 'gpt.html',
        'gpt.css': 'gpt.css',
        'gpt_rag.js': 'gpt_rag.js'
    }
    target = file_map.get(path, 'gpt.html')
    return send_from_directory(BASE_DIR, target)

# ────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # FIX 1 — Dynamically scan the pdf/ folder so ALL PDFs are always indexed,
    # including any new ones added in the future — no hardcoding needed
    pdf_dir = os.path.join(BASE_DIR, "pdf")
    PDF_PATHS = []
    if os.path.isdir(pdf_dir):
        PDF_PATHS = [
            os.path.join(pdf_dir, f)
            for f in sorted(os.listdir(pdf_dir))
            if f.lower().endswith(".pdf")
        ]
    print(f"[INFO] Found {len(PDF_PATHS)} PDF(s) in pdf/ folder:")
    for p in PDF_PATHS:
        print(f"       • {os.path.basename(p)}")

    INDEX_DIR = os.path.join(BASE_DIR, "radiology_faiss_index")

    # FIX 2 — Rebuild the index whenever the number of PDFs on disk differs
    # from what was used to build the existing index. This means adding a new
    # PDF and restarting the app is all you need — no manual folder deletion.
    needs_rebuild = False
    index_manifest = os.path.join(BASE_DIR, "radiology_faiss_index", "pdf_manifest.txt")

    if not os.path.exists(INDEX_DIR):
        print("[INFO] No index found → Building fresh index...")
        needs_rebuild = True
    else:
        # Compare current PDF list against the manifest saved at last build
        current_names = sorted([os.path.basename(p) for p in PDF_PATHS])
        if os.path.exists(index_manifest):
            with open(index_manifest, "r") as mf:
                saved_names = sorted([line.strip() for line in mf if line.strip()])
            if current_names != saved_names:
                print("[INFO] PDF folder has changed → Rebuilding index...")
                print(f"       Saved:   {saved_names}")
                print(f"       Current: {current_names}")
                needs_rebuild = True
            else:
                print("[INFO] Index is up to date → Skipping rebuild.")
        else:
            # Index exists but no manifest (old build) — rebuild to be safe
            print("[INFO] Index manifest missing → Rebuilding index...")
            needs_rebuild = True

    if needs_rebuild:
        if PDF_PATHS:
            success = create_local_index(PDF_PATHS)
            if success:
                # Save manifest so future runs can detect changes
                os.makedirs(INDEX_DIR, exist_ok=True)
                with open(index_manifest, "w") as mf:
                    mf.write("\n".join([os.path.basename(p) for p in PDF_PATHS]))
                print("[INFO] Index manifest saved.")
        else:
            print("[WARN] No PDFs found in pdf/ folder — index not built. RAG will be disabled.")

    initialize_app_services()

    APP_READY_STATE["server_ready"] = True

    print("\n" + "=" * 70)
    print("Server AI READY → http://localhost:5000")
    print("RAG (Qwen2.5), Vision (Moondream), Image Gen (SDXL)")
    print("=" * 70 + "\n")

    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)