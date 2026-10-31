// ====================================================================
// 1. GLOBAL VARIABLES AND CONFIG
// ====================================================================
const API_BASE_URL = 'http://localhost:5000';
const STORAGE_KEY = 'radiology_chat_history';
const MAX_SLOTS = 11;

// --- Core UI Elements ---
const input         = document.getElementById('text-message');
const sendButton    = document.getElementById('send-btn');
const chatbox       = document.getElementById('chatbox');
const chatSection   = document.getElementById('chat-section');
const historyList   = document.getElementById('history-list');
const newChatBtn    = document.getElementById('new-chat');
const micButton     = document.getElementById('mic-btn');
const imageUploadButton = document.getElementById('image-upload-btn');
const imageInput    = document.getElementById('image-input');
const reportButton  = document.getElementById('report-btn');
const extractButton = document.getElementById('extract-btn');
const charCounter   = document.getElementById('char-counter');
const scrollBtn     = document.getElementById('scroll-btn');
const lightbox      = document.getElementById('lightbox');
const lightboxImg   = document.getElementById('lightbox-img');
const lightboxClose = document.getElementById('lightbox-close');
const themeBtn      = document.getElementById('theme-btn');
const exportBtn     = document.getElementById('export-btn');
const copyAllBtn    = document.getElementById('copy-all-btn');
const historySearch = document.getElementById('history-search');
const toastContainer= document.getElementById('toast-container');

// --- New feature elements: patient modal, console drawer, loading screen ---
const patientModal       = document.getElementById('patient-modal');
const patientNameInput   = document.getElementById('patient-name');
const patientSexInput    = document.getElementById('patient-sex');
const patientAgeInput    = document.getElementById('patient-age');
const patientIdInput     = document.getElementById('patient-id');
const patientModalCancel = document.getElementById('patient-modal-cancel');
const patientModalConfirm= document.getElementById('patient-modal-confirm');

const consoleDrawer      = document.getElementById('console-drawer');
const consoleToggleBtn   = document.getElementById('console-toggle-btn');
const consoleCloseBtn    = document.getElementById('console-close-btn');
const consoleClearBtn    = document.getElementById('console-clear-btn');
const consoleCopyBtn = document.getElementById('console-copy-btn');
const consoleBody        = document.getElementById('console-body');

const loadingScreen      = document.getElementById('loading-screen');
const loadingBarFill     = document.getElementById('loading-bar-fill');
const stepServer         = document.getElementById('step-server');
const stepRag            = document.getElementById('step-rag');
const stepSdxl            = document.getElementById('step-sdxl');

const ctScanButton   = document.getElementById('ct-scan-btn');
const ctScanInput    = document.getElementById('ct-scan-input');
const ctProgressModal= document.getElementById('ct-progress-modal');
const ctProgressCancel = document.getElementById('ct-progress-cancel');
const ctStepUpload   = document.getElementById('ct-step-upload');
const ctStepMemory   = document.getElementById('ct-step-memory');
const ctStepSegment  = document.getElementById('ct-step-segment');
const ctStepReport   = document.getElementById('ct-step-report');

const ctChoiceModal     = document.getElementById('ct-choice-modal');
const ctChoiceFolderBtn = document.getElementById('ct-choice-folder');
const ctChoiceZipBtn    = document.getElementById('ct-choice-zip');
const ctChoiceCancelBtn = document.getElementById('ct-choice-cancel');
const ctZipInput        = document.getElementById('ct-scan-zip-input');

// --- State Variables ---
let currentSessionId = Date.now().toString();
let isNewSession = true;
let isSending = false;
let uploadedImageFile = null;
let patientInfoContext = null; // 'report' | 'ct'
let pendingPatientInfo = null;

// ====================================================================
// 2. TOAST NOTIFICATIONS
// ====================================================================
function showToast(message, type = 'success', duration = 2500) {
    const toast = document.createElement('div');
    toast.className = `toast${type !== 'success' ? ' ' + type : ''}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => toast.remove(), duration);
}

// ====================================================================
// 3. THEME TOGGLE (dark ↔ light)
// ====================================================================
function initTheme() {
    const saved = localStorage.getItem('ece_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    updateThemeIcon(saved);
}

function updateThemeIcon(theme) {
    const icon = themeBtn.querySelector('i');
    if (theme === 'dark') {
        icon.className = 'fas fa-moon';
        themeBtn.title = 'Switch to light theme';
    } else {
        icon.className = 'fas fa-sun';
        themeBtn.title = 'Switch to dark theme';
    }
}

themeBtn.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('ece_theme', next);
    updateThemeIcon(next);
    showToast(next === 'light' ? 'Light theme on' : 'Dark theme on');
});

// ====================================================================
// 4. LIGHTBOX
// ====================================================================
function openLightbox(src) {
    lightboxImg.src = src;
    lightbox.classList.add('open');
    document.body.style.overflow = 'hidden';
}

function closeLightbox() {
    lightbox.classList.remove('open');
    lightboxImg.src = '';
    document.body.style.overflow = '';
}

lightboxClose.addEventListener('click', closeLightbox);
lightbox.addEventListener('click', (e) => {
    if (e.target === lightbox) closeLightbox();
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && lightbox.classList.contains('open')) closeLightbox();
});

// ====================================================================
// 5. SCROLL-TO-BOTTOM BUTTON
// ====================================================================
function initScrollButton() {
    if (!chatSection) return;
    chatSection.addEventListener('scroll', () => {
        const distFromBottom = chatSection.scrollHeight - chatSection.scrollTop - chatSection.clientHeight;
        if (distFromBottom > 200) {
            scrollBtn.classList.add('visible');
        } else {
            scrollBtn.classList.remove('visible');
        }
    });
    scrollBtn.addEventListener('click', () => {
        chatSection.scrollTo({ top: chatSection.scrollHeight, behavior: 'smooth' });
    });
}

// ====================================================================
// 6. CHARACTER COUNTER + AUTO-RESIZE TEXTAREA
// ====================================================================
function initTextareaFeatures() {
    input.addEventListener('input', () => {
        // Auto-resize
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 160) + 'px';

        // Char counter
        const len = input.value.length;
        charCounter.textContent = `${len} char${len !== 1 ? 's' : ''}`;
        charCounter.classList.remove('warn', 'danger');
        if (len > 1500) charCounter.classList.add('danger');
        else if (len > 800) charCounter.classList.add('warn');
    });
}

// ====================================================================
// 7. EXPORT CHAT (print-to-PDF or .txt download)
// ====================================================================
exportBtn.addEventListener('click', () => {
    // Offer choice via a quick toast sequence, then download .txt
    const messages = [...chatbox.querySelectorAll('.message')];
    if (messages.length === 0) { showToast('No messages to export', 'warn'); return; }

    let text = 'ECE-LASU-LASW — Radiology AI Consultation\n';
    text += '='.repeat(50) + '\n';
    text += `Exported: ${new Date().toLocaleString()}\n`;
    text += '='.repeat(50) + '\n\n';

    messages.forEach(msg => {
        const isUser = msg.classList.contains('user-message');
        const label = isUser ? 'YOU' : 'AI';
        const content = msg.querySelector('.content')?.innerText || msg.innerText;
        text += `[${label}]\n${content.trim()}\n\n`;
    });

    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `radiology-consultation-${new Date().toISOString().slice(0,10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Conversation exported');
});

// ====================================================================
// 8. COPY ENTIRE CONVERSATION
// ====================================================================
copyAllBtn.addEventListener('click', () => {
    const messages = [...chatbox.querySelectorAll('.message')];
    if (messages.length === 0) { showToast('No messages to copy', 'warn'); return; }

    const text = messages.map(msg => {
        const isUser = msg.classList.contains('user-message');
        const label = isUser ? 'You' : 'AI';
        const content = msg.querySelector('.content')?.innerText || msg.innerText;
        return `${label}: ${content.trim()}`;
    }).join('\n\n');

    navigator.clipboard.writeText(text).then(() => {
        showToast('Conversation copied to clipboard');
    }).catch(() => {
        showToast('Copy failed — try again', 'error');
    });
});

// ====================================================================
// 9. HELPERS & RENDERING
// ====================================================================
function formatMessage(text) {
    return escapeHtml(text)
        .replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>')
        .replace(/^- (.*)$/gm, '<li>$1</li>');
}

function addCopyButton(msgDiv) {
    const btn = document.createElement('button');
    btn.className = 'copy-msg-btn';
    btn.title = 'Copy message';
    btn.innerHTML = '<i class="fas fa-clone"></i>';
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const content = msgDiv.querySelector('.content')?.innerText || msgDiv.innerText;
        navigator.clipboard.writeText(content.trim()).then(() => {
            btn.classList.add('copied');
            btn.innerHTML = '<i class="fas fa-check"></i>';
            showToast('Message copied');
            setTimeout(() => {
                btn.classList.remove('copied');
                btn.innerHTML = '<i class="fas fa-clone"></i>';
            }, 1500);
        });
    });
    msgDiv.appendChild(btn);
}

function addMessageToChatbox(content, type) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', `${type}-message`);
    msgDiv.innerHTML = `<div class="content">${formatMessage(content)}</div>`;
    addCopyButton(msgDiv);
    chatbox.appendChild(msgDiv);
    chatSection.scrollTo({ top: chatSection.scrollHeight, behavior: 'smooth' });
    return msgDiv;
}

// ── Typing indicator (replaces static "Thinking...") ──
function addTypingIndicator() {
    const div = document.createElement('div');
    div.classList.add('message', 'bot-message', 'typing-indicator');
    div.innerHTML = '<span></span><span></span><span></span>';
    div.id = 'typing-indicator';
    chatbox.appendChild(div);
    chatSection.scrollTo({ top: chatSection.scrollHeight, behavior: 'smooth' });
    return div;
}

function removeTypingIndicator() {
    const el = document.getElementById('typing-indicator');
    if (el && el.parentNode) el.parentNode.removeChild(el);
}

// ====================================================================
// 10. IMAGE DISPLAY (with lightbox instead of window.open)
// ====================================================================
function displayGeneratedImage(imageUrl) {
    let fullUrl;
    if (imageUrl.startsWith('http')) {
        fullUrl = imageUrl;
    } else {
        const cleanPath = '/' + imageUrl.replace(/^\/+/, '');
        fullUrl = `${API_BASE_URL}${cleanPath}`;
    }

    const wrapper = document.createElement('div');
    wrapper.classList.add('message', 'bot-message', 'image-container');

    const imgElement = document.createElement('img');
    imgElement.src = fullUrl;
    imgElement.alt = 'Medical Visualization';

    imgElement.onerror = () => {
        wrapper.innerHTML = '<div class="content">⚠️ Image generated but could not be displayed.</div>';
        chatbox.appendChild(wrapper);
    };

    imgElement.onload = () => {
        chatSection.scrollTo({ top: chatSection.scrollHeight, behavior: 'smooth' });
    };

    // Open in lightbox instead of new tab
    imgElement.onclick = () => openLightbox(fullUrl);

    wrapper.appendChild(imgElement);
    addCopyButton(wrapper);
    chatbox.appendChild(wrapper);
}

// ====================================================================
// 11. HISTORY & SESSION LOGIC — Persistent (disk-based via Flask API)
// ====================================================================
// PyWebView wipes localStorage on every close. History is now saved to
// chat_history.json on disk via POST /api/history and loaded via GET
// /api/history — survives restarts completely.

// In-memory cache so we don't need to await on every render
let _sessionCache = {};

async function loadSessionsFromDisk() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/history`);
        if (res.ok) {
            _sessionCache = await res.json();
        }
    } catch (e) {
        console.warn('Could not load history from disk:', e);
        _sessionCache = {};
    }
    renderHistoryUI();
}

async function saveSessionsToDisk() {
    try {
        await fetch(`${API_BASE_URL}/api/history`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(_sessionCache)
        });
    } catch (e) {
        console.warn('Could not save history to disk:', e);
    }
}

function getAllSessions() {
    return _sessionCache;
}

async function handleSessionSaving(userQuestion, botReply) {
    if (isNewSession) {
        let cleanTitle = userQuestion
            .replace(/Please generate a professional.*?:\s*/i, '')
            .replace(/Extract clinical data.*?:\s*/i, '')
            .replace(/Please generate a professional, complete radiology report.*?Notes:\n/is, 'Report: ');
        cleanTitle = cleanTitle.substring(0, 30) + (cleanTitle.length > 30 ? '...' : '');

        _sessionCache[currentSessionId] = {
            id: currentSessionId,
            title: cleanTitle || 'New Consultation',
            timestamp: Date.now(),
            html: chatbox.innerHTML
        };
        isNewSession = false;
    }

    if (_sessionCache[currentSessionId]) {
        _sessionCache[currentSessionId].html = chatbox.innerHTML;
        _sessionCache[currentSessionId].timestamp = Date.now();
    }

    // Trim to MAX_SLOTS
    const keys = Object.keys(_sessionCache).sort(
        (a, b) => _sessionCache[b].timestamp - _sessionCache[a].timestamp
    );
    if (keys.length > MAX_SLOTS) {
        for (let i = MAX_SLOTS; i < keys.length; i++) delete _sessionCache[keys[i]];
    }

    await saveSessionsToDisk();
    renderHistoryUI();
}

async function deleteSession(sessionId, e) {
    e.stopPropagation();
    delete _sessionCache[sessionId];

    if (sessionId === currentSessionId) {
        currentSessionId = Date.now().toString();
        isNewSession = true;
        chatbox.innerHTML = '<div class="message bot-message"><div class="content">New session started. How can I help?</div></div>';
    }

    await saveSessionsToDisk();
    renderHistoryUI();
    showToast('Session deleted');
}

function renderHistoryUI(filter = '') {
    const allSessions = Object.values(_sessionCache).sort((a, b) => b.timestamp - a.timestamp);
    const sessions = filter
        ? allSessions.filter(s => s.title.toLowerCase().includes(filter.toLowerCase()))
        : allSessions;

    historyList.innerHTML = '';

    if (sessions.length === 0) {
        historyList.innerHTML = `<p class="history" style="padding:8px 4px;">${filter ? 'No matches found.' : 'No History Yet...'}</p>`;
        return;
    }

    sessions.forEach(session => {
        const item = document.createElement('div');
        item.className = `history-item${session.id === currentSessionId ? ' active' : ''}`;
        item.innerHTML = `
            <div class="history-content">
                <span class="chat-title" title="${escapeHtml(session.title)}">${escapeHtml(session.title)}</span>
                <small>${new Date(session.timestamp).toLocaleDateString()} ${new Date(session.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</small>
            </div>
            <button class="history-delete-btn" title="Delete session"><i class="fas fa-trash-alt"></i></button>
        `;

        // Load session on click
        item.querySelector('.history-content').addEventListener('click', () => {
            currentSessionId = session.id;
            isNewSession = false;
            chatbox.innerHTML = session.html;
            rewireRestoredImages();
            renderHistoryUI(historySearch.value);
            chatSection.scrollTo({ top: chatSection.scrollHeight, behavior: 'smooth' });
        });

        // Rename on double-click
        const titleEl = item.querySelector('.chat-title');
        titleEl.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            titleEl.contentEditable = 'true';
            titleEl.focus();
            const range = document.createRange();
            range.selectNodeContents(titleEl);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
        });
        titleEl.addEventListener('blur', async () => {
            titleEl.contentEditable = 'false';
            const newTitle = titleEl.textContent.trim().substring(0, 40) || session.title;
            if (_sessionCache[session.id]) {
                _sessionCache[session.id].title = newTitle;
                await saveSessionsToDisk();
                showToast('Session renamed');
            }
        });
        titleEl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); titleEl.blur(); }
            if (e.key === 'Escape') { titleEl.textContent = session.title; titleEl.blur(); }
        });

        // Delete button
        item.querySelector('.history-delete-btn').addEventListener('click',
            (e) => deleteSession(session.id, e));

        historyList.appendChild(item);
    });
}

// Re-attach lightbox click on images restored from localStorage HTML
function rewireRestoredImages() {
    chatbox.querySelectorAll('.image-container img').forEach(img => {
        img.style.cursor = 'zoom-in';
        img.onclick = () => openLightbox(img.src);
    });
}

// ====================================================================
// 12. CORE CHAT LOGIC
// ====================================================================
imageUploadButton.addEventListener('click', () => imageInput.click());

imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        uploadedImageFile = file;
        const msg = addMessageToChatbox(`📎 Ready to upload: ${file.name}`, 'user');
        msg.style.opacity = '0.7';
    }
});

async function sendMessage(userInput = null, contextPrefix = '') {
    if (isSending) return;

    const rawText = userInput || input.value.trim();
    if (!rawText && !uploadedImageFile) return;

    isSending = true;
    let finalMessage = contextPrefix + rawText;

    if (rawText) addMessageToChatbox(rawText, 'user');
    input.value = '';
    input.style.height = 'auto';
    charCounter.textContent = '0 chars';
    charCounter.classList.remove('warn', 'danger');

    const typingDiv = addTypingIndicator();

    try {
        // --- 1. Handle Image Upload ---
        if (uploadedImageFile) {
            typingDiv.innerHTML = '<div class="content" style="font-size:13px;color:var(--text-muted)">Uploading medical scan…</div>';
            const formData = new FormData();
            formData.append('file', uploadedImageFile);

            const uploadResponse = await fetch(`${API_BASE_URL}/api/upload`, {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) throw new Error('Image upload failed');
            const uploadData = await uploadResponse.json();

            uploadedImageFile = null;
            imageInput.value = '';

            finalMessage += `\n\n[SYSTEM: The user has uploaded an image for analysis at this path: ${uploadData.filepath}]`;
        }

        // --- 2. Request AI Response ---
        const response = await fetch(`${API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: finalMessage })
        });

        if (!response.ok) throw new Error(`Server error: ${response.status}`);
        const data = await response.json();

        removeTypingIndicator();

        let botReply = data.reply || 'No response.';
        let imageUrl = data.image_url || null;

        // Fallback regex if backend doesn't return image_url
        if (!imageUrl) {
            const markdownImageRegex = /!\[.*?\]\((https?:\/\/[^\s)]+\.(?:png|jpg|jpeg|gif))\)/i;
            const rawUrlRegex = /(https?:\/\/[^\s)]+\.(?:png|jpg|jpeg|gif))/i;
            const relativePathRegex = /(?:^|\s)(\/output\/[\w\-]+\.(?:png|jpg|jpeg|gif))/i;
            const bareRelativeRegex = /output\/([\w\-]+\.(?:png|jpg|jpeg|gif))/i;

            let match = botReply.match(markdownImageRegex);
            if (match) { imageUrl = match[1]; botReply = botReply.replace(match[0], '').trim(); }
            else {
                match = botReply.match(rawUrlRegex);
                if (match) { imageUrl = match[0]; botReply = botReply.replace(imageUrl, '').trim(); }
                else {
                    match = botReply.match(relativePathRegex);
                    if (match) { imageUrl = match[1]; botReply = botReply.replace(match[0], '').trim(); }
                    else {
                        match = botReply.match(bareRelativeRegex);
                        if (match) { imageUrl = `/output/${match[1]}`; botReply = botReply.replace(match[0], '').trim(); }
                    }
                }
            }
        }

        if (imageUrl) {
            botReply = botReply
                .replace(/\/output\/[\w\-]+\.(?:png|jpg|jpeg|gif)/gi, '')
                .replace(/output\/[\w\-]+\.(?:png|jpg|jpeg|gif)/gi, '')
                .trim();
        }

        if (botReply) addMessageToChatbox(botReply, 'bot');
        if (imageUrl) displayGeneratedImage(imageUrl);

        handleSessionSaving(rawText, botReply);

    } catch (error) {
        removeTypingIndicator();
        console.error('Critical Error:', error);
        addMessageToChatbox(`Error: ${error.message}`, 'bot');
        showToast('Request failed — check server', 'error');
    } finally {
        isSending = false;
        setTimeout(() => input.focus(), 10);
    }
}

// ====================================================================
// 13. EVENT LISTENERS
// ====================================================================
function initEventListeners() {
    // New Chat
    newChatBtn.addEventListener('click', (e) => {
        e.preventDefault();
        currentSessionId = Date.now().toString();
        isNewSession = true;
        chatbox.innerHTML = '<div class="message bot-message"><div class="content">New session started. How can I help?</div></div>';
        renderHistoryUI(historySearch.value);
        showToast('New session started');
    });

    // Send Button
    sendButton.addEventListener('click', (e) => {
        e.preventDefault();
        sendMessage();
    });

    // Enter Key
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Report Button — opens the patient info modal instead of a disconnected prompt()
    reportButton.addEventListener('click', (e) => {
        e.preventDefault();
        if (!input.value.trim()) return addMessageToChatbox('Enter findings first.', 'bot');
        patientInfoContext = 'report';
        openPatientModal();
    });

    // Extract Button
    extractButton.addEventListener('click', (e) => {
        e.preventDefault();
        if (!input.value.trim()) return addMessageToChatbox('Enter text first.', 'bot');
        sendMessage(input.value.trim(), 'Extract clinical data to JSON from: ');
    });

    // Mobile Menu
    const menuBtn = document.getElementById('menu-btn');
    const chatHistory = document.getElementById('chat-history');
    if (menuBtn && chatHistory) {
        menuBtn.addEventListener('click', (e) => {
            chatHistory.classList.toggle('show');
            e.stopPropagation();
        });
        document.addEventListener('click', (event) => {
            if (window.innerWidth <= 768 && !chatHistory.contains(event.target) && !menuBtn.contains(event.target)) {
                chatHistory.classList.remove('show');
            }
        });
    }

    // History search filter
    historySearch.addEventListener('input', () => {
        renderHistoryUI(historySearch.value);
    });
}

// ====================================================================
// 14. SPEECH RECOGNITION
// ====================================================================
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.continuous = false;
        recognition.onresult = (event) => {
            input.value = event.results[0][0].transcript;
            // Trigger resize + counter update
            input.dispatchEvent(new Event('input'));
            micButton.classList.remove('active');
            showToast('Voice captured');
        };
        recognition.onend = () => micButton.classList.remove('active');
        micButton.addEventListener('click', () => {
            if (micButton.classList.contains('active')) recognition.stop();
            else { micButton.classList.add('active'); recognition.start(); }
        });
    } else {
        if (micButton) micButton.style.display = 'none';
    }
}

// ====================================================================
// 15. PATIENT INFO MODAL (fixes the report generation bug)
// ====================================================================
// The previous implementation sent the findings and the patient info as
// two SEPARATE, disconnected chat messages 100ms apart — the AI generated
// the report before it ever received the patient's name/age. This modal
// collects everything up front and sends ONE combined prompt instead.

function openPatientModal() {
    patientNameInput.value = '';
    patientSexInput.value = '';
    patientAgeInput.value = '';
    patientIdInput.value = '';
    patientModal.classList.add('open');
    setTimeout(() => patientNameInput.focus(), 50);
}

function closePatientModal() {
    patientModal.classList.remove('open');
}

function submitPatientModal() {
    const findings = input.value.trim();
    const name = patientNameInput.value.trim();
    const sex = patientSexInput.value;
    const age = patientAgeInput.value.trim();
    const caseId = patientIdInput.value.trim();

    if (!name || !sex || !age) {
        showToast('Please fill in name, sex, and age', 'warn');
        return;
    }

    // Build ONE combined, structured prompt — this is the actual fix.
    // Patient info and findings now travel together in a single message,
    // so the AI has everything it needs in one shot.
    let patientLine = `Patient: ${name}, ${sex}, ${age} years old`;
    if (caseId) patientLine += ` (Case ID: ${caseId})`;

    const combinedPrompt =
        `Please generate a professional, complete radiology report.\n\n` +
        `${patientLine}\n\n` +
        `Clinical Findings / Notes:\n${findings}\n\n` +
        `Structure the report with: Patient Information, Clinical History, ` +
        `Findings, Impression, and Recommendations.`;

    closePatientModal();

    if (patientInfoContext === 'ct') {
        pendingPatientInfo = { name, sex, age, caseId };
        ctChoiceModal.classList.add('open');
    } else {
        sendMessage(combinedPrompt);
    }
    patientInfoContext = null;
}

patientModalCancel.addEventListener('click', closePatientModal);
patientModalConfirm.addEventListener('click', submitPatientModal);
patientModal.addEventListener('click', (e) => {
    if (e.target === patientModal) closePatientModal();
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && patientModal.classList.contains('open')) closePatientModal();
});
// Allow Enter key to move between fields / submit from the last field
patientIdInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); submitPatientModal(); }
});

// ====================================================================
// 16. CT SCAN — 3D volumetric analysis flow
// ====================================================================
ctScanButton.addEventListener('click', () => {
    patientInfoContext = 'ct';
    openPatientModal();
});

ctChoiceFolderBtn.addEventListener('click', () => {
    ctChoiceModal.classList.remove('open');
    ctScanInput.click();
});

ctChoiceZipBtn.addEventListener('click', () => {
    ctChoiceModal.classList.remove('open');
    ctZipInput.click();
});

ctChoiceCancelBtn.addEventListener('click', () => {
    ctChoiceModal.classList.remove('open');
});

function setCtStep(stepEl, state) {
    stepEl.classList.remove('active', 'done');
    stepEl.classList.add(state);
}

function resetCtProgressUI() {
    [ctStepUpload, ctStepMemory, ctStepSegment, ctStepReport].forEach(el => {
        el.classList.remove('active', 'done');
    });
    setCtStep(ctStepUpload, 'active');
}

async function handleCtUpload(files) {
    if (files.length === 0) return;

    const isZip = files.length === 1 && files[0].name.toLowerCase().endsWith('.zip');
    const label = isZip
        ? `📦 Ready to upload CT scan: ${files[0].name}`
        : `📁 Ready to upload CT scan: ${files.length} file(s)`;
    const msg = addMessageToChatbox(label, 'user');
    msg.style.opacity = '0.7';

    resetCtProgressUI();
    ctProgressModal.classList.add('open');
    ctProgressCancel.disabled = true;

    try {
        const formData = new FormData();
        files.forEach(f => formData.append('files', f));

        const uploadRes = await fetch(`${API_BASE_URL}/api/upload_ct`, {
            method: 'POST',
            body: formData
        });
        if (!uploadRes.ok) throw new Error('CT upload failed');
        const uploadData = await uploadRes.json();
        setCtStep(ctStepUpload, 'done');

        const analyzeRes = await fetch(`${API_BASE_URL}/api/analyze_ct`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dicom_folder: uploadData.dicom_folder, patient_info: pendingPatientInfo || {} })
        });
        if (!analyzeRes.ok) throw new Error('Could not start CT analysis');
        const { job_id } = await analyzeRes.json();

        pollCtStatus(job_id);

    } catch (err) {
        ctProgressModal.classList.remove('open');
        ctProgressCancel.disabled = false;
        showToast(err.message || 'CT analysis failed to start', 'error');
    }
}

ctScanInput.addEventListener('change', (e) => {
    handleCtUpload(Array.from(e.target.files));
    e.target.value = '';
});

ctZipInput.addEventListener('change', (e) => {
    handleCtUpload(Array.from(e.target.files));
    e.target.value = '';
});

function pollCtStatus(jobId) {
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/api/ct_status/${jobId}`);
            if (!res.ok) return;
            const job = await res.json();

            if (job.stage === 'freeing_memory') setCtStep(ctStepMemory, 'active');
            if (job.stage === 'segmenting') {
                setCtStep(ctStepMemory, 'done');
                setCtStep(ctStepSegment, 'active');
            }
            if (job.stage === 'generating_report' || job.stage === 'reloading_sdxl') {
                setCtStep(ctStepSegment, 'done');
                setCtStep(ctStepReport, 'active');
            }
            if (job.stage === 'done') {
                setCtStep(ctStepReport, 'done');
                clearInterval(interval);
                setTimeout(() => {
                    ctProgressModal.classList.remove('open');
                    ctProgressCancel.disabled = false;
                    addMessageToChatbox(job.report, 'bot');
                    handleSessionSaving('CT Volumetric Analysis', job.report);
                    pendingPatientInfo = null;
                    showToast('CT report generated');
                }, 500);
            }
            if (job.stage === 'error') {
                clearInterval(interval);
                ctProgressModal.classList.remove('open');
                ctProgressCancel.disabled = false;
                showToast(`CT analysis failed: ${job.error}`, 'error');
            }
        } catch (e) {
            // server unreachable momentarily — keep polling
        }
    }, 2000);
}

ctProgressCancel.addEventListener('click', () => {
    if (!ctProgressCancel.disabled) ctProgressModal.classList.remove('open');
});

// ====================================================================
// 17. CONSOLE DRAWER (replaces the visible terminal window)
// ====================================================================
// app.py captures every print() into an in-memory buffer and exposes it
// via GET /api/logs. This polls that endpoint and renders it here, so the
// .exe can run with console=False (no terminal window) while still giving
// you full visibility when you need it.

let consoleLastCount = 0;
let consolePollInterval = null;
let consoleRawLines = [];

function renderConsoleLogs(logs) {
    if (logs.length === 0) return;
    if (consoleBody.querySelector('.console-empty')) consoleBody.innerHTML = '';

    const newLogs = logs.slice(consoleLastCount);
    newLogs.forEach(log => {
        const line = document.createElement('div');
        line.className = 'console-line';
        if (/error|failed|traceback/i.test(log.text)) line.classList.add('error');
        else if (/warn/i.test(log.text)) line.classList.add('warn');
        line.innerHTML = `<span class="console-time">${log.time}</span>${escapeHtml(log.text)}`;
        consoleBody.appendChild(line);
        consoleRawLines.push(`[${log.time}] ${log.text}`);
    });
    consoleLastCount = logs.length;

    // Auto-scroll only if already near the bottom
    const nearBottom = consoleBody.scrollHeight - consoleBody.scrollTop - consoleBody.clientHeight < 80;
    if (nearBottom) consoleBody.scrollTop = consoleBody.scrollHeight;

    // Flag the toggle button if a new error appeared and drawer is closed
    if (!consoleDrawer.classList.contains('open') && newLogs.some(l => /error|failed|traceback/i.test(l.text))) {
        consoleToggleBtn.classList.add('has-error');
    }
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

async function pollConsoleLogs() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/logs`);
        if (!res.ok) return;
        const data = await res.json();
        renderConsoleLogs(data.logs || []);
    } catch (e) {
        // Server not reachable yet — silently retry on next interval
    }
}

function initConsoleDrawer() {
    consoleToggleBtn.addEventListener('click', () => {
        consoleDrawer.classList.toggle('open');
        consoleToggleBtn.classList.remove('has-error');
    });
    consoleCopyBtn.addEventListener('click', () => {
        const text = consoleRawLines.join('\n');
        navigator.clipboard.writeText(text).then(() => {
            showToast('Console logs copied.');
        }).catch(() => {
            showToast('Copy failed.');
        });
    });
    consoleCloseBtn.addEventListener('click', () => consoleDrawer.classList.remove('open'));
    consoleClearBtn.addEventListener('click', () => {
        consoleBody.innerHTML = '<p class="console-empty">Display cleared.</p>';
        consoleRawLines = [];
    });

    pollConsoleLogs();
    consolePollInterval = setInterval(pollConsoleLogs, 2000);
}

// ====================================================================
// 18. LOADING SCREEN (replaces the visible terminal as the "is it ready"
//     signal — polls /api/ready until the server, RAG, and SDXL are up)
// ====================================================================
function setStepState(stepEl, state) {
    // state: 'active' | 'done'
    stepEl.classList.remove('active', 'done');
    stepEl.classList.add(state);
}

async function pollReadyState() {
    let attempts = 0;
    const maxAttempts = 360; // ~6 minutes at 1s intervals — generous for first-run model loads

    setStepState(stepServer, 'active');

    const interval = setInterval(async () => {
        attempts++;
        try {
            const res = await fetch(`${API_BASE_URL}/api/ready`);
            if (res.ok) {
                const data = await res.json();

                if (data.server_ready) {
                    setStepState(stepServer, 'done');
                    loadingBarFill.style.width = '35%';
                }
                if (data.rag_ready) {
                    setStepState(stepRag, 'done');
                    loadingBarFill.style.width = '70%';
                } else if (data.server_ready) {
                    setStepState(stepRag, 'active');
                }
                if (data.sdxl_attempted) {
                    setStepState(stepSdxl, 'done');
                    loadingBarFill.style.width = '100%';
                } else if (data.rag_ready) {
                    setStepState(stepSdxl, 'active');
                }

                // Server + RAG are enough to start using the app; SDXL can
                // finish loading quietly in the background.
                if (data.server_ready && data.rag_ready) {
                    clearInterval(interval);
                    setTimeout(() => {
                        loadingScreen.classList.add('hidden');
                    }, 400);
                }
            }
        } catch (e) {
            // Backend not up yet — keep polling
        }

        if (attempts >= maxAttempts) {
            clearInterval(interval);
            loadingScreen.classList.add('hidden');
            showToast('Taking longer than expected — continuing anyway', 'warn');
        }
    }, 1000);
}

// ====================================================================
// 19. INITIALIZATION
// ====================================================================
document.addEventListener('DOMContentLoaded', function () {
    initTheme();
    loadSessionsFromDisk();   // Loads history from disk, then calls renderHistoryUI()
    initEventListeners();
    initSpeechRecognition();
    initScrollButton();
    initTextareaFeatures();
    initConsoleDrawer();
    pollReadyState();
});