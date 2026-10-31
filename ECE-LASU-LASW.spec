# ECE-LASU-LASW.spec
# ---------------------------------------------------------
# PyInstaller build spec for Expert-Level Radiology AI
#
# Usage:
#   pip install pyinstaller
#   pyinstaller ECE-LASU-LASW.spec
#
# Output: dist\ECE-LASU-LASW\  (folder with .exe + all deps)
# ---------------------------------------------------------

block_cipher = None

a = Analysis(
    ['launcher.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # ── Frontend files ──────────────────────────────
        ('gpt.html',    '.'),
        ('gpt.css',     '.'),
        ('gpt_rag.js',  '.'),
        ('app.py',      '.'),
        ('run_totalsegmentator.py', '.'),
        ('TotalSegRunner', 'TotalSegRunner'),
        ('Profile.png', '.'),
        ('Profile.ico', '.'),

        # ── Package metadata ────────────────────────────
        ('.venv/Lib/site-packages/requests-*.dist-info', '.'),
        ('.venv/Lib/site-packages/diffusers-*.dist-info', '.'),
        ('.venv/Lib/site-packages/transformers-*.dist-info', '.'),
        ('.venv/Lib/site-packages/torch-*.dist-info', '.'),
        ('.venv/Lib/site-packages/accelerate-*.dist-info', '.'),
        ('.venv/Lib/site-packages/safetensors-*.dist-info', '.'),
        ('.venv/Lib/site-packages/huggingface_hub-*.dist-info', '.'),
        ('.venv/Lib/site-packages/filelock-*.dist-info', '.'),
        ('.venv/Lib/site-packages/packaging-*.dist-info', '.'),
        ('.venv/Lib/site-packages/numpy-*.dist-info', '.'),
        ('.venv/Lib/site-packages/Pillow-*.dist-info', '.'),
        ('.venv/Lib/site-packages/webview', 'webview'),

        # ── LangChain packages ──────────────────────────
        ('.venv/Lib/site-packages/langchain_classic', 'langchain_classic'),
        ('.venv/Lib/site-packages/langchain_core', 'langchain_core'),
        ('.venv/Lib/site-packages/langchain_community', 'langchain_community'),
        ('.venv/Lib/site-packages/langchain_ollama', 'langchain_ollama'),
        ('.venv/Lib/site-packages/langchain_text_splitters', 'langchain_text_splitters'),
        ('.venv/Lib/site-packages/langgraph', 'langgraph'),
        ('.venv/Lib/site-packages/diffusers', 'diffusers'),
        ('.venv/Lib/site-packages/transformers', 'transformers'),

        # ── LangChain metadata ──────────────────────────
        ('.venv/Lib/site-packages/langchain_classic-1.0.7.dist-info', 'langchain_classic-1.0.7.dist-info'),
        ('.venv/Lib/site-packages/langchain_core-1.4.0.dist-info', 'langchain_core-1.4.0.dist-info'),
        ('.venv/Lib/site-packages/langchain_community-0.4.2.dist-info', 'langchain_community-0.4.2.dist-info'),
        ('.venv/Lib/site-packages/langchain_ollama-1.1.0.dist-info', 'langchain_ollama-1.1.0.dist-info'),
        ('.venv/Lib/site-packages/langchain_text_splitters-1.1.2.dist-info', 'langchain_text_splitters-1.1.2.dist-info'),

        # ── Font Awesome ──────────────────────────
        ('fontawesome/css/all.min.css', 'fontawesome/css'),
        ('fontawesome/webfonts/*',      'fontawesome/webfonts'),

        # ── Include app.py so launcher can import it ────
        # (PyInstaller compiles it, but we list it explicitly
        #  so n imports are picked up correctly)
    ],
    hiddenimports=[
        # Flask ecosystem
        'flask',
        'flask_cors',
        'werkzeug',
        'werkzeug.utils',
        'werkzeug.serving',
        'jinja2',
        'click',

        # LangChain
        'langchain',
        'langchain_core',
        'langchain_core.tools',
        'langchain_core.messages',
        'langchain_core.prompts',
        'langchain_community',
        'langchain_community.document_loaders',
        'langchain_community.vectorstores',
        'langchain_community.vectorstores.faiss',
        'langchain_text_splitters',
        'langchain_ollama',
        'langchain_classic',
        'langchain_classic.chains',
        'langchain_classic.chains.retrieval',
        'langchain_classic.chains.retrieval_qa',
        'langchain_classic.chains.retrieval_qa.base',
        'langchain_classic.chains.combine_documents',
        'langchain_classic.chains.combine_documents.stuff',
        'langchain_classic.chains.question_answering',
        'langchain_classic.chains.question_answering.stuff_prompt',
        'langgraph.checkpoint',
        'langgraph.prebuilt',
        'langgraph.graph.state',

        # LangGraph
        'langgraph',
        'langgraph.graph',

        # FAISS
        'faiss',

        # Diffusers / Torch
        'diffusers',
        'diffusers.pipelines',
        'diffusers.pipelines.stable_diffusion_xl',
        'diffusers.schedulers',
        'torch',
        'torchvision',
        'transformers',
        'accelerate',
        'safetensors',

        # Image handling
        'PIL',
        'PIL.Image',

        # Misc
        'typing_extensions',
        'pydantic',
        'pydantic.v1',
        'numpy',
        'scipy',
        'regex',
        'tqdm',
        'huggingface_hub',
        'pypdf',
        'urllib3',
        'certifi',
        'charset_normalizer',
        'idna',
        'requests',
        'aiohttp',
        'anyio',
        'sniffio',

        #unittest
        'unittest',
        'unittest.mock',

        #email
        'email',
        'email.mime',
        'email.mime.text',
        'email.mime.multipart',
        'email.encoders',

        #others
        'http',
        'http.server',
        'http.client',
        'http.cookiejar',
        'xmlrpc',
        'xmlrpc.client',
        'xmlrpc.server',
        'webview',
        'webview.platforms.winforms',
        'clr',
        'pydicom',
        'nibabel',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Trim what we definitely don't need to reduce size
        'tkinter',
        'turtle',
        'curses',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ECE-LASU-LASW',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,           # Compress — set False if UPX is not installed
    console=False,       # Shows a terminal window (useful for seeing errors) if True
                        # Change to False once everything works reliably
    icon='Profile.ico', # Optional: your app icon (.ico preferred on Windows)
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ECE-LASU-LASW',
)