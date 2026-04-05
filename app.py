import os
import uuid
import base64
import requests
import traceback
import operator
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from typing import TypedDict, Annotated, List

# Cloud-Friendly AI Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AnyMessage

# LangGraph Imports
from langgraph.graph import StateGraph, END

# Standardized Chain Imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ====================================================================
# 1. APP CONFIG & KEYS
# ====================================================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Use Environment Variables for safety (or paste keys here for testing)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_key_here")
STABILITY_KEY = os.environ.get("STABILITY_API_KEY", "your_stability_key_here")

# Globals
RAG_CHAIN_OBJ = None
AGENT_WORKFLOW = None

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ====================================================================
# 2. TOOL 1: IMAGE GENERATION (Cloud API)
# ====================================================================
@tool
def generate_image(prompt: str) -> str:
    """
    Generates a medical visualization or anatomy mock-up via Stability AI.
    """
    if not STABILITY_KEY:
        return "Error: Stability API key missing."

    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }
    
    body = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "steps": 30,
    }

    try:
        response = requests.post(url, headers=headers, json=body)
        if response.status_code != 200:
            return f"Stability API Error: {response.text}"

        data = response.json()
        filename = f"gen_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(OUTPUT_FOLDER, filename)

        with open(filepath, "wb") as f:
            f.write(base64.b64decode(data["artifacts"][0]["base64"]))

        # On Render, we use a relative path so the frontend can find it
        return f"/output/{filename}"
    except Exception as e:
        return f"Image generation failed: {str(e)}"

# ====================================================================
# 3. TOOL 2: VISION ANALYSIS (Cloud API)
# ====================================================================
@tool
def analyze_radiology_image(image_path: str, query: str = "Describe this medical image") -> str:
    """
    Analyzes medical scans (X-ray, CT) using Llama-3-Vision via Groq.
    """
    if not os.path.exists(image_path):
        return f"Error: Image not found at {image_path}"

    try:
        # Groq Llama-3 Vision model
        vision_llm = ChatGroq(model="llama-3.2-11b-vision-preview", groq_api_key=GROQ_API_KEY)

        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ]
        )
        
        response = vision_llm.invoke([message])
        return f"Vision Analysis: {response.content}"
    except Exception as e:
        return f"Vision analysis failed: {str(e)}"

# ====================================================================
# 4. TOOL 3: RAG KNOWLEDGE BASE
# ====================================================================
def create_rag_chain_object(vectorstore):
    # Using Llama-3-70B via Groq - Super accurate for medical reports
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    SYSTEM_PROMPT = (
        "You are WANIKO, an expert radiologist. Assist with clinical reports and findings. "
        "Use provided context to be precise. If you generate a report, use professional terminology. "
        "\n\nContext:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

@tool
def radiology_rag_tool(question: str) -> str:
    """Answers medical questions using the indexed radiology knowledge base."""
    global RAG_CHAIN_OBJ
    if not RAG_CHAIN_OBJ: return "Knowledge base not ready."
    try:
        return RAG_CHAIN_OBJ.invoke({"input": question})["answer"]
    except Exception as e: return f"RAG error: {str(e)}"

# Register Tools
AGENT_TOOLS = [radiology_rag_tool, generate_image, analyze_radiology_image]

# ====================================================================
# 5. AGENT LOGIC (LangGraph)
# ====================================================================
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

def call_model(state: AgentState):
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
    llm_with_tools = llm.bind_tools(AGENT_TOOLS)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def call_tool(state: AgentState):
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", [])
    results = []
    for t_call in tool_calls:
        func = next((t for t in AGENT_TOOLS if t.name == t_call["name"]), None)
        obs = func.invoke(t_call["args"]) if func else "Unknown tool"
        results.append(ToolMessage(content=str(obs), tool_call_id=t_call["id"]))
    return {"messages": results}

def setup_agent(vectorstore):
    global RAG_CHAIN_OBJ
    RAG_CHAIN_OBJ = create_rag_chain_object(vectorstore)
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("call_tool", call_tool)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", lambda x: "call_tool" if x["messages"][-1].tool_calls else END)
    workflow.add_edge("call_tool", "agent")
    return workflow.compile()

# ====================================================================
# 6. INITIALIZATION (Relative Paths for Cloud)
# ====================================================================
def initialize_services():
    global AGENT_WORKFLOW
    INDEX_PATH = "radiology_faiss_index"
    
    # Use CPU-friendly embeddings (No local Ollama needed)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(INDEX_PATH):
        try:
            vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            AGENT_WORKFLOW = setup_agent(vectorstore)
            print(">> Cloud-connected Agent Initialized.")
        except Exception as e: print(f">> Init Error: {e}")
    else:
        print("!! Index folder missing. Please run indexing first.")

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file:
        filename = secure_filename(f"{uuid.uuid4().hex[:6]}_{file.filename}")
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        return jsonify({'message': 'Uploaded', 'filepath': path}), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    msg = data.get('message', '').strip()
    if not AGENT_WORKFLOW: return jsonify({'reply': 'Server initializing...'}), 503
    try:
        res = AGENT_WORKFLOW.invoke({"messages": [HumanMessage(content=msg)]})
        return jsonify({'reply': res["messages"][-1].content})
    except Exception as e: return jsonify({'reply': str(e)}), 500

@app.route('/output/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    # This serves your HTML from the root folder
    if path == '' or path == 'index.html': return send_from_directory('.', 'index.html')
    return send_from_directory('.', path)

if __name__ == '__main__':
    initialize_services()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
