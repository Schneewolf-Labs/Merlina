// Inference Module - Handles model loading and chat for testing trained models

import { MerlinaAPI, WS_URL } from './api.js';
import { Toast } from './ui.js';

class InferenceManager {
    constructor() {
        this.toast = new Toast();
        this.conversationHistory = [];
        this.isGenerating = false;
        this.streamSocket = null;
        this.localModels = [];

        this.setupEventListeners();
        this.loadLocalModels();
        this.checkModelStatus();
    }

    setupEventListeners() {
        // Source toggle
        const sourceSelect = document.getElementById('inference-source');
        if (sourceSelect) {
            sourceSelect.addEventListener('change', () => this.onSourceChange());
        }

        // Trained model select — show base model info
        const trainedSelect = document.getElementById('inference-trained-model');
        if (trainedSelect) {
            trainedSelect.addEventListener('change', () => this.onTrainedModelChange());
        }

        // Load / Unload buttons
        document.getElementById('inference-load-btn')?.addEventListener('click', () => this.loadModel());
        document.getElementById('inference-unload-btn')?.addEventListener('click', () => this.unloadModel());

        // Chat controls
        document.getElementById('inference-send-btn')?.addEventListener('click', () => this.sendMessage());
        document.getElementById('inference-clear-btn')?.addEventListener('click', () => this.clearChat());

        // Enter to send (Shift+Enter for newline)
        document.getElementById('inference-chat-input')?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
    }

    onSourceChange() {
        const source = document.getElementById('inference-source').value;
        const trainedGroup = document.getElementById('inference-trained-group');
        const manualGroup = document.getElementById('inference-manual-group');
        const hfTokenGroup = document.getElementById('inference-hf-token-group');

        if (source === 'trained') {
            trainedGroup.style.display = 'block';
            manualGroup.style.display = 'none';
            hfTokenGroup.style.display = 'none';
        } else if (source === 'huggingface') {
            trainedGroup.style.display = 'none';
            manualGroup.style.display = 'block';
            hfTokenGroup.style.display = 'block';
            document.getElementById('inference-model-name').placeholder = 'meta-llama/Meta-Llama-3-8B-Instruct';
        } else {
            // local path
            trainedGroup.style.display = 'none';
            manualGroup.style.display = 'block';
            hfTokenGroup.style.display = 'none';
            document.getElementById('inference-model-name').placeholder = '/home/user/models/my-model';
        }
    }

    onTrainedModelChange() {
        const select = document.getElementById('inference-trained-model');
        const info = document.getElementById('inference-trained-info');
        const model = this.localModels.find(m => m.name === select.value);
        if (model && model.is_lora && info) {
            info.textContent = `LoRA adapter — base model: ${model.base_model}`;
        } else if (info) {
            info.textContent = '';
        }
    }

    async loadLocalModels() {
        try {
            const data = await MerlinaAPI.listInferenceModels();
            this.localModels = data.models || [];

            const select = document.getElementById('inference-trained-model');
            if (!select) return;

            select.innerHTML = '';

            if (this.localModels.length === 0) {
                select.innerHTML = '<option value="">No trained models found</option>';
                return;
            }

            for (const model of this.localModels) {
                // Primary option = transformers backend (base + LoRA or full model).
                const opt = document.createElement('option');
                opt.value = model.name;
                opt.dataset.backend = 'transformers';
                const suffix = model.is_lora ? ` (LoRA on ${model.base_model})` : '';
                opt.textContent = model.name + suffix;
                select.appendChild(opt);

                // One entry per GGUF quant surfaced via the manifest.
                if (Array.isArray(model.gguf) && model.gguf.length > 0) {
                    for (const gguf of model.gguf) {
                        const ggufOpt = document.createElement('option');
                        // Encode backend + quant into the value so onTrainedModelChange
                        // and getSelectedModelName can reconstruct the payload.
                        ggufOpt.value = `${model.name}::gguf::${gguf.quant_type}`;
                        ggufOpt.dataset.backend = 'llama_cpp';
                        ggufOpt.dataset.modelName = model.name;
                        ggufOpt.dataset.ggufPath = gguf.path;
                        ggufOpt.dataset.ggufQuantType = gguf.quant_type;
                        const sizeMb = gguf.size_bytes ? ` — ${(gguf.size_bytes / (1024**3)).toFixed(1)} GB` : '';
                        ggufOpt.textContent = `  └─ ${gguf.quant_type} (GGUF)${sizeMb}`;
                        select.appendChild(ggufOpt);
                    }
                }
            }

            this.onTrainedModelChange();
        } catch (error) {
            console.error('Failed to load local models:', error);
        }
    }

    async checkModelStatus() {
        try {
            const status = await MerlinaAPI.getInferenceStatus();
            if (status.loaded) {
                this.showLoadedState(status);
            }
        } catch (error) {
            console.error('Failed to check inference status:', error);
        }
    }

    getSelectedModelName() {
        const source = document.getElementById('inference-source').value;
        if (source === 'trained') {
            return document.getElementById('inference-trained-model').value;
        }
        return document.getElementById('inference-model-name').value.trim();
    }

    async loadModel() {
        const payload = this.buildLoadPayload();
        if (!payload) {
            this.toast.error('Please select or enter a model name');
            return;
        }

        const loadBtn = document.getElementById('inference-load-btn');
        loadBtn.disabled = true;
        loadBtn.textContent = payload.backend === 'llama_cpp'
            ? '🔮 Spinning up llama-server...'
            : '⏳ Loading model...';

        try {
            const result = await MerlinaAPI.loadInferenceModel(payload);
            this.showLoadedState(result);
            this.toast.success(`Model loaded: ${result.model_name}`);
        } catch (error) {
            this.toast.error(`Failed to load model: ${error.message}`);
            console.error('Model load error:', error);
        } finally {
            loadBtn.disabled = false;
            loadBtn.textContent = '🧙 Load Model';
        }
    }

    /**
     * Build the /inference/load payload based on the selected option.
     * Detects GGUF entries by the data-backend attribute set in loadLocalModels().
     */
    buildLoadPayload() {
        const source = document.getElementById('inference-source').value;
        const hfToken = document.getElementById('inference-hf-token').value || null;

        if (source === 'trained') {
            const select = document.getElementById('inference-trained-model');
            if (!select || !select.selectedOptions || select.selectedOptions.length === 0) {
                return null;
            }
            const opt = select.selectedOptions[0];
            const backend = opt.dataset.backend || 'transformers';

            if (backend === 'llama_cpp') {
                return {
                    model_name: opt.dataset.modelName,
                    backend: 'llama_cpp',
                    gguf_path: opt.dataset.ggufPath || null,
                    gguf_quant_type: opt.dataset.ggufQuantType || null,
                    hf_token: hfToken,
                };
            }

            return {
                model_name: opt.value,
                backend: 'transformers',
                use_4bit: document.getElementById('inference-4bit').checked,
                hf_token: hfToken,
            };
        }

        // Free-form entry (HF model ID or local path) — always transformers.
        const modelName = document.getElementById('inference-model-name').value.trim();
        if (!modelName) return null;
        return {
            model_name: modelName,
            backend: 'transformers',
            use_4bit: document.getElementById('inference-4bit').checked,
            hf_token: hfToken,
        };
    }

    showLoadedState(info) {
        // Show status badge
        const statusDiv = document.getElementById('inference-model-status');
        const backend = info.backend || 'transformers';

        let subtitle;
        if (backend === 'llama_cpp') {
            const quant = info.gguf_quant_type ? ` — ${info.gguf_quant_type}` : '';
            const server = info.server_url ? ` · ${info.server_url}` : '';
            subtitle = `GGUF via llama-server${quant}${server}`;
        } else {
            const loraInfo = info.is_lora ? ` (LoRA on ${info.base_model})` : '';
            const quantInfo = info.use_4bit ? '4-bit' : 'full precision';
            const memInfo = info.gpu_memory ? ` — ${info.gpu_memory} VRAM` : '';
            subtitle = `${quantInfo}${memInfo}${loraInfo}`;
        }

        statusDiv.innerHTML = `
            <div style="padding: 15px; background: #e8f5e9; border-radius: 10px; border: 1px solid #4caf50;">
                <div style="font-weight: bold; color: #2e7d32; margin-bottom: 5px;">
                    ✓ Model Loaded ${backend === 'llama_cpp' ? '🔮' : ''}
                </div>
                <div style="font-size: 0.9em; color: #555;">
                    <strong>${info.model_name}</strong><br>
                    ${subtitle}
                </div>
            </div>`;
        statusDiv.style.display = 'block';

        // Show unload button, settings, and chat
        document.getElementById('inference-unload-btn').style.display = 'block';
        document.getElementById('inference-settings').style.display = 'block';
        document.getElementById('inference-chat-section').style.display = 'block';

        // Focus chat input
        document.getElementById('inference-chat-input')?.focus();
    }

    showUnloadedState() {
        document.getElementById('inference-model-status').style.display = 'none';
        document.getElementById('inference-unload-btn').style.display = 'none';
        document.getElementById('inference-settings').style.display = 'none';
        document.getElementById('inference-chat-section').style.display = 'none';
    }

    async unloadModel() {
        const unloadBtn = document.getElementById('inference-unload-btn');
        unloadBtn.disabled = true;

        try {
            await MerlinaAPI.unloadInferenceModel();
            this.showUnloadedState();
            this.clearChat();
            this.toast.success('Model unloaded');
        } catch (error) {
            this.toast.error(`Failed to unload: ${error.message}`);
        } finally {
            unloadBtn.disabled = false;
        }
    }

    getGenerationParams() {
        return {
            max_new_tokens: parseInt(document.getElementById('inference-max-tokens').value) || 512,
            temperature: parseFloat(document.getElementById('inference-temperature').value) || 0.7,
            top_p: parseFloat(document.getElementById('inference-top-p').value) || 0.9,
            top_k: parseInt(document.getElementById('inference-top-k').value) || 50,
            repetition_penalty: parseFloat(document.getElementById('inference-rep-penalty').value) || 1.1,
            do_sample: document.getElementById('inference-do-sample').checked,
        };
    }

    buildMessages(userMessage) {
        const messages = [];
        const systemPrompt = document.getElementById('inference-system-prompt')?.value?.trim();
        if (systemPrompt) {
            messages.push({ role: 'system', content: systemPrompt });
        }

        // Add conversation history
        for (const msg of this.conversationHistory) {
            messages.push(msg);
        }

        // Add current user message
        messages.push({ role: 'user', content: userMessage });
        return messages;
    }

    async sendMessage() {
        if (this.isGenerating) return;

        const input = document.getElementById('inference-chat-input');
        const message = input.value.trim();
        if (!message) return;

        input.value = '';
        this.isGenerating = true;

        const sendBtn = document.getElementById('inference-send-btn');
        sendBtn.disabled = true;
        sendBtn.textContent = '...';

        // Add user message to UI
        this.appendMessage('user', message);

        // Build messages array (buildMessages appends the user message itself,
        // so don't add to conversationHistory until we get the response)
        const messages = this.buildMessages(message);

        const useStreaming = document.getElementById('inference-stream')?.checked;

        try {
            if (useStreaming) {
                await this.streamResponse(messages);
            } else {
                await this.chatResponse(messages);
            }
        } catch (error) {
            this.appendMessage('error', `Error: ${error.message}`);
        } finally {
            this.isGenerating = false;
            sendBtn.disabled = false;
            sendBtn.textContent = 'Send';
            input.focus();
        }
    }

    async chatResponse(messages) {
        const params = this.getGenerationParams();
        const result = await MerlinaAPI.inferenceChat({
            messages,
            ...params,
        });

        this.appendMessage('assistant', result.response);
        this.conversationHistory.push({ role: 'user', content: messages[messages.length - 1].content });
        this.conversationHistory.push({ role: 'assistant', content: result.response });
    }

    async streamResponse(messages) {
        const params = this.getGenerationParams();

        return new Promise((resolve, reject) => {
            // Create a placeholder message element for streaming
            const msgEl = this.appendMessage('assistant', '');
            const contentEl = msgEl.querySelector('.inference-msg-content');

            let fullResponse = '';
            const ws = new WebSocket(`${WS_URL}/ws-inference`);

            ws.onopen = () => {
                ws.send(JSON.stringify({ messages, ...params }));
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'token') {
                    fullResponse += data.text;
                    contentEl.textContent = fullResponse;
                    // Auto-scroll
                    const container = document.getElementById('inference-chat-messages');
                    container.scrollTop = container.scrollHeight;
                } else if (data.type === 'done') {
                    this.conversationHistory.push({ role: 'user', content: messages[messages.length - 1].content });
                    this.conversationHistory.push({ role: 'assistant', content: fullResponse });
                    ws.close();
                    resolve();
                } else if (data.type === 'error') {
                    contentEl.textContent = `Error: ${data.message}`;
                    contentEl.style.color = '#dc2626';
                    ws.close();
                    reject(new Error(data.message));
                }
            };

            ws.onerror = () => {
                reject(new Error('WebSocket connection failed'));
            };

            ws.onclose = (event) => {
                if (!event.wasClean && !fullResponse) {
                    reject(new Error('Connection closed unexpectedly'));
                }
            };
        });
    }

    appendMessage(role, content) {
        const container = document.getElementById('inference-chat-messages');
        const msgEl = document.createElement('div');
        msgEl.className = `inference-msg inference-msg-${role}`;

        const roleLabel = role === 'user' ? 'You' : role === 'assistant' ? 'Model' : 'System';
        const roleIcon = role === 'user' ? '👤' : role === 'assistant' ? '🤖' : '⚠️';

        msgEl.innerHTML = `
            <div class="inference-msg-header">
                <span>${roleIcon} ${roleLabel}</span>
            </div>
            <div class="inference-msg-content">${this.escapeHtml(content)}</div>
        `;

        container.appendChild(msgEl);
        container.scrollTop = container.scrollHeight;
        return msgEl;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    clearChat() {
        this.conversationHistory = [];
        const container = document.getElementById('inference-chat-messages');
        if (container) container.innerHTML = '';
    }
}

export { InferenceManager };
