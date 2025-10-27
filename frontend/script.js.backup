// Merlina Frontend JavaScript

// Dynamically detect API URL from current page location
// This allows the app to work on any domain (localhost, production, reverse proxy)
const API_URL = '';  // Empty string = relative URLs (same origin)
const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;

let activeJobs = {};
let currentJobId = null;
let pollInterval = null;
let uploadedDatasetId = null; // Store uploaded dataset ID
let datasetColumns = null; // Store dataset columns for mapping
let datasetSamples = null; // Store sample data

// Log configuration for debugging
console.log('🔧 Merlina Configuration:');
console.log(`  API URL: ${API_URL || window.location.origin} (relative)`);
console.log(`  WebSocket URL: ${WS_URL}`);

// DOM Elements
const form = document.getElementById('training-form');
const jobsSection = document.getElementById('jobs-section');
const jobsContainer = document.getElementById('jobs-container');
const modal = document.getElementById('job-modal');
const closeModal = document.querySelector('.close');
const toast = document.getElementById('toast');

// Form Elements
const pushHub = document.getElementById('push-hub');
const useWandb = document.getElementById('use-wandb');

// Dataset Elements
const datasetSourceType = document.getElementById('dataset-source-type');
const hfSourceConfig = document.getElementById('hf-source-config');
const uploadSourceConfig = document.getElementById('upload-source-config');
const localSourceConfig = document.getElementById('local-source-config');
const datasetFormatType = document.getElementById('dataset-format-type');
const customFormatConfig = document.getElementById('custom-format-config');
const uploadButton = document.getElementById('upload-button');
const previewButton = document.getElementById('preview-dataset-button');
const previewFormattedButton = document.getElementById('preview-formatted-button');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Load any active jobs
    loadJobs();

    // Set up event listeners
    form.addEventListener('submit', handleSubmit);
    closeModal.addEventListener('click', () => closeJobModal());
    document.getElementById('stop-button').addEventListener('click', stopTraining);

    // Dataset source type change
    datasetSourceType.addEventListener('change', (e) => {
        // Hide all source configs
        hfSourceConfig.style.display = 'none';
        uploadSourceConfig.style.display = 'none';
        localSourceConfig.style.display = 'none';

        // Show selected config
        const sourceType = e.target.value;
        if (sourceType === 'huggingface') {
            hfSourceConfig.style.display = 'block';
        } else if (sourceType === 'upload') {
            uploadSourceConfig.style.display = 'block';
        } else if (sourceType === 'local_file') {
            localSourceConfig.style.display = 'block';
        }
    });

    // Dataset format type change
    const qwen3FormatConfig = document.getElementById('qwen3-format-config');
    datasetFormatType.addEventListener('change', (e) => {
        const formatType = e.target.value;
        customFormatConfig.style.display = formatType === 'custom' ? 'block' : 'none';
        qwen3FormatConfig.style.display = formatType === 'qwen3' ? 'block' : 'none';
    });

    // Upload dataset button
    uploadButton.addEventListener('click', handleDatasetUpload);

    // Preview dataset buttons
    previewButton.addEventListener('click', handleDatasetPreview);
    previewFormattedButton.addEventListener('click', handleFormattedPreview);

    // Model preload button
    const preloadModelButton = document.getElementById('preload-model-button');
    console.log('Preload button element:', preloadModelButton);
    if (preloadModelButton) {
        preloadModelButton.addEventListener('click', handleModelPreload);
        console.log('Preload button event listener attached');
    } else {
        console.error('Preload button not found!');
    }

    // Column inspection button
    const inspectColumnsButton = document.getElementById('inspect-columns-button');
    if (inspectColumnsButton) {
        inspectColumnsButton.addEventListener('click', handleInspectColumns);
    }

    // GPU refresh button
    const refreshGpuButton = document.getElementById('refresh-gpu-button');
    if (refreshGpuButton) {
        refreshGpuButton.addEventListener('click', handleRefreshGPUs);
    }

    // Advanced settings toggle
    const toggleAdvanced = document.getElementById('toggle-advanced');
    const advancedSections = document.querySelectorAll('.advanced-section');
    let advancedVisible = false;

    toggleAdvanced.addEventListener('click', () => {
        advancedVisible = !advancedVisible;
        advancedSections.forEach(section => {
            section.style.display = advancedVisible ? 'block' : 'none';
        });
        toggleAdvanced.querySelector('span').textContent =
            advancedVisible ? '⚙️ Hide Advanced Settings' : '⚙️ Show Advanced Settings';
    });

    // LoRA configuration toggle
    const useLora = document.getElementById('use-lora');
    const loraSettings = document.getElementById('lora-settings');
    const mergeLoraConfig = document.getElementById('merge-lora-config');

    useLora.addEventListener('change', (e) => {
        loraSettings.style.display = e.target.checked ? 'block' : 'none';
        // Also update merge LoRA visibility
        if (mergeLoraConfig) {
            mergeLoraConfig.style.display = e.target.checked ? 'block' : 'none';
        }
    });

    // Initialize LoRA config visibility based on checkbox state
    if (useLora.checked) {
        loraSettings.style.display = 'block';
        if (mergeLoraConfig) {
            mergeLoraConfig.style.display = 'block';
        }
    } else {
        if (mergeLoraConfig) {
            mergeLoraConfig.style.display = 'none';
        }
    }

    // W&B configuration toggle
    const wandbConfig = document.getElementById('wandb-config');
    useWandb.addEventListener('change', (e) => {
        wandbConfig.style.display = e.target.checked ? 'block' : 'none';
    });

    // Initialize W&B config visibility based on checkbox state
    if (useWandb.checked) {
        wandbConfig.style.display = 'block';
    }

    // HuggingFace Hub configuration toggle
    const hfHubConfig = document.getElementById('hf-hub-config');
    pushHub.addEventListener('change', (e) => {
        hfHubConfig.style.display = e.target.checked ? 'block' : 'none';
    });

    // Initialize HF Hub config visibility based on checkbox state
    if (pushHub.checked) {
        hfHubConfig.style.display = 'block';
    }

    // Clear all jobs button handler
    const clearAllJobsBtn = document.getElementById('clear-all-jobs-btn');
    if (clearAllJobsBtn) {
        clearAllJobsBtn.addEventListener('click', clearAllJobs);
    }

    // Close modal when clicking outside
    window.addEventListener('click', (e) => {
        if (e.target === modal) closeJobModal();
    });
});

// Handle dataset upload
async function handleDatasetUpload() {
    const fileInput = document.getElementById('dataset-file');
    const file = fileInput.files[0];

    if (!file) {
        showToast('⚠️ Please select a file to upload', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        uploadButton.disabled = true;
        uploadButton.textContent = '⏳ Uploading...';

        const response = await fetch(`${API_URL}/dataset/upload-file`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error(`Upload failed: ${response.status}`);

        const data = await response.json();
        uploadedDatasetId = data.dataset_id;

        document.getElementById('upload-status').innerHTML = `
            <div style="padding: 10px; background: var(--success); color: white; border-radius: 5px; margin-top: 10px;">
                ✅ Uploaded: ${data.filename} (ID: ${data.dataset_id})
            </div>
        `;

        showToast(`✅ Dataset uploaded successfully!`, 'success');

    } catch (error) {
        showToast(`❌ Upload failed: ${error.message}`, 'error');
        document.getElementById('upload-status').innerHTML = `
            <div style="padding: 10px; background: var(--danger); color: white; border-radius: 5px; margin-top: 10px;">
                ❌ Upload failed: ${error.message}
            </div>
        `;
    } finally {
        uploadButton.disabled = false;
        uploadButton.textContent = '📤 Upload Dataset';
    }
}

// Handle model preload
async function handleModelPreload() {
    console.log('handleModelPreload called');
    const baseModel = document.getElementById('base-model').value.trim();
    const hfToken = document.getElementById('hf-token-preload').value.trim();
    const preloadButton = document.getElementById('preload-model-button');
    const modelStatus = document.getElementById('model-status');
    const modelInfo = document.getElementById('model-info');

    console.log('Base model:', baseModel);
    console.log('HF token:', hfToken ? '[PRESENT]' : '[EMPTY]');

    if (!baseModel) {
        showToast('⚠️ Please enter a base model name', 'error');
        return;
    }

    try {
        preloadButton.disabled = true;
        preloadButton.textContent = '⏳ Loading model tokenizer...';
        modelStatus.style.display = 'none';

        const response = await fetch(`${API_URL}/model/preload`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_name: baseModel,
                hf_token: hfToken || null
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to preload model');
        }

        const data = await response.json();

        // Display model info
        modelInfo.innerHTML = `
            <strong>${data.model_name}</strong><br/>
            Vocab Size: ${data.vocab_size.toLocaleString()}<br/>
            Max Length: ${data.model_max_length.toLocaleString()}<br/>
            Chat Template: ${data.has_chat_template ? '✓ Detected' : '✗ Not found'}<br/>
            ${data.has_chat_template ? '<span style="color: var(--primary-purple);">💡 You can now use "Tokenizer" format for accurate preview!</span>' : ''}
        `;
        modelStatus.style.display = 'block';

        showToast(`✅ Model tokenizer loaded successfully!`, 'success');

    } catch (error) {
        showToast(`❌ Failed to load model: ${error.message}`, 'error');
        modelInfo.innerHTML = `<span style="color: var(--danger);">${error.message}</span>`;
        modelStatus.querySelector('div').style.background = '#ffebee';
        modelStatus.querySelector('div').style.borderColor = 'var(--danger)';
        modelStatus.querySelector('div div:first-child').textContent = '✗ Error';
        modelStatus.querySelector('div div:first-child').style.color = 'var(--danger)';
        modelStatus.style.display = 'block';
    } finally {
        preloadButton.disabled = false;
        preloadButton.textContent = '🔮 Validate & Preload Model';
    }
}

// Handle column inspection
async function handleInspectColumns() {
    const inspectButton = document.getElementById('inspect-columns-button');
    const columnMappingConfig = document.getElementById('column-mapping-config');

    try {
        inspectButton.disabled = true;
        inspectButton.textContent = '⏳ Loading columns...';

        const datasetConfig = getDatasetSourceConfig();

        const response = await fetch(`${API_URL}/dataset/columns`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(datasetConfig)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to fetch columns');
        }

        const data = await response.json();
        datasetColumns = data.columns;
        datasetSamples = data.samples;

        // Populate column mapping dropdowns
        populateColumnMappings(data.columns);

        // Show available columns
        document.getElementById('available-columns').textContent = data.columns.join(', ');

        // Show sample data
        if (data.samples && data.samples.length > 0) {
            const samplePreview = document.getElementById('column-sample-preview');
            const sampleContent = document.getElementById('column-sample-content');
            sampleContent.textContent = JSON.stringify(data.samples[0], null, 2);
            samplePreview.style.display = 'block';
        }

        // Show column mapping UI
        columnMappingConfig.style.display = 'block';

        showToast(`✅ Found ${data.columns.length} columns in dataset`, 'success');

    } catch (error) {
        showToast(`❌ Failed to inspect columns: ${error.message}`, 'error');
    } finally {
        inspectButton.disabled = false;
        inspectButton.textContent = '🔍 Inspect Dataset Columns';
    }
}

// Helper to get dataset source config (without format)
function getDatasetSourceConfig() {
    const sourceType = datasetSourceType.value;

    let source = { source_type: sourceType };

    if (sourceType === 'huggingface') {
        source.repo_id = document.getElementById('hf-repo-id').value;
        source.split = document.getElementById('hf-split').value;
    } else if (sourceType === 'upload') {
        if (!uploadedDatasetId) {
            throw new Error('Please upload a dataset first');
        }
        source.dataset_id = uploadedDatasetId;
    } else if (sourceType === 'local_file') {
        source.file_path = document.getElementById('local-file-path').value;
        const format = document.getElementById('local-file-format').value;
        if (format) source.file_format = format;
    }

    return {
        source: source,
        format: { format_type: 'chatml' },  // Dummy format, not used for column inspection
        test_size: 0.1
    };
}

// Populate column mapping dropdowns
function populateColumnMappings(columns) {
    const selects = [
        'map-prompt',
        'map-chosen',
        'map-rejected',
        'map-system',
        'map-reasoning'
    ];

    selects.forEach(selectId => {
        const select = document.getElementById(selectId);
        // Clear existing options except first
        while (select.options.length > 1) {
            select.remove(1);
        }

        // Add column options
        columns.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            select.appendChild(option);
        });

        // Auto-select if column name matches
        const targetColumn = selectId.replace('map-', '');
        if (columns.includes(targetColumn)) {
            select.value = targetColumn;
        }
    });
}

// Handle dataset preview
async function handleDatasetPreview() {
    try {
        previewButton.disabled = true;
        previewButton.textContent = '⏳ Loading...';

        const datasetConfig = getDatasetConfig();

        const response = await fetch(`${API_URL}/dataset/preview`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(datasetConfig)
        });

        if (!response.ok) throw new Error(`Preview failed: ${response.status}`);

        const data = await response.json();

        // Display preview
        const previewDiv = document.getElementById('dataset-preview');
        const previewContent = document.getElementById('dataset-preview-content');
        const formattedPreview = document.getElementById('formatted-preview');

        previewContent.textContent = JSON.stringify(data.samples, null, 2);
        previewDiv.style.display = 'block';
        formattedPreview.style.display = 'none'; // Hide formatted preview

        showToast(`✅ Loaded ${data.num_samples} sample(s)`, 'success');

    } catch (error) {
        showToast(`❌ Preview failed: ${error.message}`, 'error');
    } finally {
        previewButton.disabled = false;
        previewButton.textContent = '🔍 Preview Raw Data';
    }
}

// Handle formatted preview
async function handleFormattedPreview() {
    try {
        previewFormattedButton.disabled = true;
        previewFormattedButton.textContent = '⏳ Loading...';

        const datasetConfig = getDatasetConfig();

        const response = await fetch(`${API_URL}/dataset/preview-formatted`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(datasetConfig)
        });

        if (!response.ok) throw new Error(`Preview failed: ${response.status}`);

        const data = await response.json();

        if (data.samples.length === 0) {
            throw new Error('No samples returned');
        }

        // Get first sample
        const sample = data.samples[0];

        // Display formatted preview
        const formattedPreview = document.getElementById('formatted-preview');
        const rawPreview = document.getElementById('dataset-preview');

        document.getElementById('formatted-prompt').textContent = sample.prompt;
        document.getElementById('formatted-chosen').textContent = sample.chosen;
        document.getElementById('formatted-rejected').textContent = sample.rejected;

        // Display format type
        const formatType = datasetConfig.format.format_type;
        const formatNames = {
            'tokenizer': 'Tokenizer (Model Native)',
            'chatml': 'ChatML',
            'llama3': 'Llama 3',
            'mistral': 'Mistral Instruct',
            'qwen3': `Qwen 3 (thinking ${datasetConfig.format.enable_thinking ? 'enabled' : 'disabled'})`,
            'custom': 'Custom Template'
        };
        document.getElementById('format-type-display').textContent = formatNames[formatType] || formatType;

        formattedPreview.style.display = 'block';
        rawPreview.style.display = 'none'; // Hide raw preview

        showToast(`✅ Preview formatted with ${formatNames[formatType]}`, 'success');

    } catch (error) {
        showToast(`❌ Formatted preview failed: ${error.message}`, 'error');
    } finally {
        previewFormattedButton.disabled = false;
        previewFormattedButton.textContent = '✨ Preview Formatted';
    }
}

// Get dataset configuration from form
function getDatasetConfig() {
    const sourceType = datasetSourceType.value;
    const formatType = datasetFormatType.value;

    // Build source config
    let source = { source_type: sourceType };

    if (sourceType === 'huggingface') {
        source.repo_id = document.getElementById('hf-repo-id').value;
        source.split = document.getElementById('hf-split').value;
    } else if (sourceType === 'upload') {
        if (!uploadedDatasetId) {
            throw new Error('Please upload a dataset first');
        }
        source.dataset_id = uploadedDatasetId;
    } else if (sourceType === 'local_file') {
        source.file_path = document.getElementById('local-file-path').value;
        const format = document.getElementById('local-file-format').value;
        if (format) source.file_format = format;
    }

    // Build format config
    let format = { format_type: formatType };

    if (formatType === 'custom') {
        format.custom_templates = {
            prompt_template: document.getElementById('custom-prompt-template').value,
            chosen_template: document.getElementById('custom-chosen-template').value,
            rejected_template: document.getElementById('custom-rejected-template').value
        };
    }

    if (formatType === 'qwen3') {
        format.enable_thinking = document.getElementById('enable-thinking').checked;
    }

    // Build full config
    const config = {
        source: source,
        format: format,
        test_size: parseFloat(document.getElementById('test-size').value)
    };

    const maxSamples = document.getElementById('max-samples').value;
    if (maxSamples) {
        config.max_samples = parseInt(maxSamples);
    }

    // Add model name for tokenizer format preview
    const baseModel = document.getElementById('base-model').value.trim();
    if (baseModel) {
        config.model_name = baseModel;
    }

    // Add column mapping if configured
    const columnMapping = getColumnMapping();
    if (columnMapping && Object.keys(columnMapping).length > 0) {
        config.column_mapping = columnMapping;
    }

    return config;
}

// Get column mapping from UI
function getColumnMapping() {
    const mapping = {};

    const promptCol = document.getElementById('map-prompt').value;
    const chosenCol = document.getElementById('map-chosen').value;
    const rejectedCol = document.getElementById('map-rejected').value;
    const systemCol = document.getElementById('map-system').value;
    const reasoningCol = document.getElementById('map-reasoning').value;

    if (promptCol) mapping[promptCol] = 'prompt';
    if (chosenCol) mapping[chosenCol] = 'chosen';
    if (rejectedCol) mapping[rejectedCol] = 'rejected';
    if (systemCol) mapping[systemCol] = 'system';
    if (reasoningCol) mapping[reasoningCol] = 'reasoning';

    return mapping;
}

// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();

    // Parse target modules from comma-separated string
    const targetModulesStr = document.getElementById('target-modules').value;
    const targetModules = targetModulesStr.split(',').map(s => s.trim()).filter(s => s.length > 0);

    // Get dataset configuration
    let datasetConfig;
    try {
        datasetConfig = getDatasetConfig();
    } catch (error) {
        showToast(`❌ Dataset config error: ${error.message}`, 'error');
        return;
    }

    // Gather form data
    const config = {
        base_model: document.getElementById('base-model').value,
        output_name: document.getElementById('output-name').value,

        // Dataset configuration
        dataset: datasetConfig,

        // LoRA settings
        use_lora: document.getElementById('use-lora').checked,
        lora_r: parseInt(document.getElementById('lora-r').value),
        lora_alpha: parseInt(document.getElementById('lora-alpha').value),
        lora_dropout: parseFloat(document.getElementById('lora-dropout').value),
        target_modules: targetModules,

        // Training settings
        learning_rate: parseFloat(document.getElementById('learning-rate').value),
        num_epochs: parseInt(document.getElementById('epochs').value),
        batch_size: parseInt(document.getElementById('batch-size').value),
        gradient_accumulation_steps: parseInt(document.getElementById('grad-accum').value),
        max_length: parseInt(document.getElementById('max-length').value),
        max_prompt_length: parseInt(document.getElementById('max-prompt-length').value),
        beta: parseFloat(document.getElementById('beta').value),

        // Priority 1 settings
        seed: parseInt(document.getElementById('seed').value),
        max_grad_norm: parseFloat(document.getElementById('max-grad-norm').value),

        // Advanced training settings
        warmup_ratio: parseFloat(document.getElementById('warmup-ratio').value),
        eval_steps: parseFloat(document.getElementById('eval-steps').value),

        // Priority 2 advanced settings
        shuffle_dataset: document.getElementById('shuffle-dataset').checked,
        weight_decay: parseFloat(document.getElementById('weight-decay').value),
        lr_scheduler_type: document.getElementById('lr-scheduler-type').value,
        gradient_checkpointing: document.getElementById('gradient-checkpointing').checked,
        logging_steps: parseInt(document.getElementById('logging-steps').value),

        // Optimizer settings
        optimizer_type: document.getElementById('optimizer-type').value,
        adam_beta1: parseFloat(document.getElementById('adam-beta1').value),
        adam_beta2: parseFloat(document.getElementById('adam-beta2').value),
        adam_epsilon: parseFloat(document.getElementById('adam-epsilon').value),

        // Attention settings
        attn_implementation: document.getElementById('attn-implementation').value,

        // GPU settings
        gpu_ids: getSelectedGPUs(),

        // Options
        use_4bit: document.getElementById('use-4bit').checked,
        use_wandb: document.getElementById('use-wandb').checked,
        push_to_hub: document.getElementById('push-hub').checked,
        merge_lora_before_upload: document.getElementById('merge-lora-before-upload').checked,
        hf_hub_private: document.getElementById('hf-hub-private').checked,

        // API Keys
        wandb_key: document.getElementById('wandb-key').value || null,
        hf_token: document.getElementById('hf-token').value || document.getElementById('hf-token-preload').value || null,

        // W&B settings
        wandb_project: document.getElementById('wandb-project').value || null,
        wandb_run_name: document.getElementById('wandb-run-name').value || null,
        wandb_tags: document.getElementById('wandb-tags').value ?
            document.getElementById('wandb-tags').value.split(',').map(tag => tag.trim()).filter(tag => tag) :
            null,
        wandb_notes: document.getElementById('wandb-notes').value || null
    };
    
    // Disable form
    setFormEnabled(false);
    
    try {
        const response = await fetch(`${API_URL}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        
        // Show success message
        showToast(`✨ Training spell cast! Job ID: ${data.job_id}`, 'success');
        
        // Add job to tracking
        activeJobs[data.job_id] = {
            name: config.output_name,
            status: 'started',
            config: config
        };
        
        // Update UI
        loadJobs();
        
        // Open monitoring modal
        showJobDetails(data.job_id);

        // Only clear the model name field (preserve other settings)
        document.getElementById('output-name').value = '';
        
    } catch (error) {
        showToast(`❌ Failed to cast spell: ${error.message}`, 'error');
    } finally {
        setFormEnabled(true);
    }
}

// Load and display jobs
async function loadJobs() {
    try {
        const response = await fetch(`${API_URL}/jobs`);
        const jobs = await response.json();
        
        if (Object.keys(jobs).length > 0) {
            jobsSection.style.display = 'block';
            jobsContainer.innerHTML = '';
            
            for (const [jobId, job] of Object.entries(jobs)) {
                // Update our local tracking
                if (activeJobs[jobId]) {
                    activeJobs[jobId].status = job.status;
                    activeJobs[jobId].progress = job.progress;
                }
                
                // Create job card
                const jobCard = createJobCard(jobId, job);
                jobsContainer.appendChild(jobCard);
            }
        } else {
            jobsSection.style.display = 'none';
        }
    } catch (error) {
        console.error('Failed to load jobs:', error);
    }
}

// Create job card element
function createJobCard(jobId, job) {
    const card = document.createElement('div');
    card.className = 'job-card';

    const statusClass = `status-${job.status}`;
    const progressPercent = Math.round((job.progress || 0) * 100);

    card.innerHTML = `
        <div class="job-card-header">
            <div style="flex: 1; cursor: pointer;" onclick="event.stopPropagation(); showJobDetails('${jobId}')">
                <h4>${activeJobs[jobId]?.name || jobId}</h4>
                <span class="job-status-badge ${statusClass}">${job.status}</span>
            </div>
            <button class="delete-job-btn" onclick="event.stopPropagation(); deleteJob('${jobId}')" title="Delete job">
                🗑️
            </button>
        </div>
        <div class="progress-bar" style="cursor: pointer;" onclick="showJobDetails('${jobId}')">
            <div class="progress-fill" style="width: ${progressPercent}%"></div>
            <span class="progress-text">${progressPercent}%</span>
        </div>
    `;

    return card;
}

// Show job details modal
async function showJobDetails(jobId) {
    currentJobId = jobId;
    modal.style.display = 'block';
    
    // Update modal header
    document.getElementById('job-name').textContent = activeJobs[jobId]?.name || jobId;
    document.getElementById('job-id').textContent = jobId;
    
    // Start polling for updates
    updateJobStatus();
    pollInterval = setInterval(updateJobStatus, 3000); // Poll every 3 seconds
}

// Update job status in modal
async function updateJobStatus() {
    if (!currentJobId) return;
    
    try {
        const response = await fetch(`${API_URL}/status/${currentJobId}`);
        const status = await response.json();
        
        // Update progress
        const progressPercent = Math.round((status.progress || 0) * 100);
        document.getElementById('progress-fill').style.width = `${progressPercent}%`;
        document.getElementById('progress-text').textContent = `${progressPercent}%`;
        
        // Update status text
        document.getElementById('status-text').textContent = `Status: ${status.status}`;
        document.getElementById('job-status').textContent = status.status;
        
        // Update metrics
        document.getElementById('current-step').textContent = status.current_step || '-';
        document.getElementById('loss-value').textContent = status.loss ? status.loss.toFixed(4) : '-';

        // Update GPU memory if available
        const gpuMemoryEl = document.getElementById('gpu-memory-value');
        if (gpuMemoryEl) {
            if (status.gpu_memory !== undefined && status.gpu_memory !== null) {
                gpuMemoryEl.textContent = `${status.gpu_memory.toFixed(2)} GB`;
            } else {
                gpuMemoryEl.textContent = '-';
            }
        }

        // Show W&B link if using wandb and URL is available
        const wandbLink = document.getElementById('wandb-link');
        if (activeJobs[currentJobId]?.config?.use_wandb && status.wandb_url) {
            wandbLink.style.display = 'block';
            wandbLink.href = status.wandb_url;
        } else {
            wandbLink.style.display = 'none';
        }
        
        // Handle job status changes
        const stopButton = document.getElementById('stop-button');

        if (status.status === 'completed') {
            clearInterval(pollInterval);
            showToast('✅ Training completed successfully!', 'success');
            stopButton.disabled = true;
        } else if (status.status === 'failed') {
            clearInterval(pollInterval);
            showToast(`❌ Training failed: ${status.error || 'Unknown error'}`, 'error');
            stopButton.disabled = true;
        } else if (status.status === 'stopped') {
            clearInterval(pollInterval);
            const finalStep = status.current_step || '?';
            showToast(`⏸️ Training stopped gracefully at step ${finalStep}`, 'warning');
            stopButton.disabled = true;
        } else if (status.status === 'stopping') {
            stopButton.disabled = true;
            stopButton.textContent = '⏸️ Stopping...';
        } else if (status.status === 'training' || status.status === 'initializing' || status.status === 'loading_model' || status.status === 'loading_dataset') {
            // Re-enable stop button for active statuses (in case it was disabled)
            stopButton.disabled = false;
            stopButton.textContent = '⏸️ Stop Training';
        }
        
        // Update jobs list
        loadJobs();
        
    } catch (error) {
        console.error('Failed to update job status:', error);
    }
}

// Close job modal
function closeJobModal() {
    modal.style.display = 'none';
    currentJobId = null;
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
    document.getElementById('stop-button').disabled = false;
}

// Stop training gracefully
async function stopTraining() {
    if (!currentJobId) return;

    if (confirm('⚠️ Stop training gracefully?\n\nThe training will:\n• Complete the current step\n• Save a checkpoint\n• Exit cleanly\n\nThis cannot be undone.')) {
        try {
            const response = await fetch(`${API_URL}/jobs/${currentJobId}/stop`, {
                method: 'POST'
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to stop training');
            }

            const data = await response.json();

            if (data.status === 'success') {
                showToast('✅ Stop request sent. Training will stop after current step.', 'success');

                // Disable stop button to prevent multiple clicks
                const stopBtn = document.getElementById('stop-button');
                if (stopBtn) {
                    stopBtn.disabled = true;
                    stopBtn.textContent = '⏸️ Stopping...';
                }
            } else {
                showToast(`⚠️ ${data.message}`, 'warning');
            }
        } catch (error) {
            console.error('Failed to stop training:', error);
            showToast(`❌ Failed to stop training: ${error.message}`, 'error');
        }
    }
}

// Delete a specific job
async function deleteJob(jobId) {
    if (!confirm(`Are you sure you want to delete job ${jobId}?`)) {
        return;
    }

    try {
        const response = await fetch(`${API_URL}/jobs/${jobId}`, {
            method: 'DELETE'
        });

        if (!response.ok) throw new Error('Failed to delete job');

        const data = await response.json();
        showToast(`✅ ${data.message}`, 'success');

        // Remove from active jobs
        delete activeJobs[jobId];

        // Close modal if this job was open
        if (currentJobId === jobId) {
            closeJobModal();
        }

        // Reload jobs list
        loadJobs();

    } catch (error) {
        console.error('Failed to delete job:', error);
        showToast(`❌ Failed to delete job: ${error.message}`, 'error');
    }
}

// Clear all jobs
async function clearAllJobs() {
    if (!confirm('⚠️ Are you sure you want to delete ALL jobs? This action cannot be undone!')) {
        return;
    }

    try {
        const response = await fetch(`${API_URL}/jobs`, {
            method: 'DELETE'
        });

        if (!response.ok) throw new Error('Failed to clear jobs');

        const data = await response.json();
        showToast(`✅ ${data.message}`, 'success');

        // Clear active jobs
        Object.keys(activeJobs).forEach(key => delete activeJobs[key]);

        // Close modal
        closeJobModal();

        // Reload jobs list (will hide section since no jobs)
        loadJobs();

    } catch (error) {
        console.error('Failed to clear jobs:', error);
        showToast(`❌ Failed to clear jobs: ${error.message}`, 'error');
    }
}

// Show toast notification
function showToast(message, type = 'info') {
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 5000);
}

// Enable/disable form
function setFormEnabled(enabled) {
    const inputs = form.querySelectorAll('input, select, button');
    inputs.forEach(input => {
        input.disabled = !enabled;
    });
    
    if (!enabled) {
        document.querySelector('.button-icon').style.animation = 'rotate 1s linear infinite';
    } else {
        document.querySelector('.button-icon').style.animation = 'none';
    }
}

// Add some magical interactions
document.addEventListener('DOMContentLoaded', () => {
    // Add sparkle effect on input focus
    const inputs = document.querySelectorAll('.magic-input, .magic-select');
    inputs.forEach(input => {
        input.addEventListener('focus', (e) => {
            createSparkle(e.target);
        });
    });
    
    // Animate wizard hat on hover
    const wizardHat = document.querySelector('.wizard-hat');
    wizardHat.addEventListener('mouseenter', () => {
        wizardHat.style.animation = 'wiggle 0.5s ease-in-out';
        setTimeout(() => {
            wizardHat.style.animation = 'wiggle 2s ease-in-out infinite';
        }, 500);
    });
});

// Create sparkle effect
function createSparkle(element) {
    const sparkle = document.createElement('div');
    sparkle.style.cssText = `
        position: absolute;
        pointer-events: none;
        width: 4px;
        height: 4px;
        background: #ffd700;
        border-radius: 50%;
        animation: sparkleFloat 1s ease-out forwards;
    `;
    
    const rect = element.getBoundingClientRect();
    sparkle.style.left = `${rect.left + Math.random() * rect.width}px`;
    sparkle.style.top = `${rect.top + rect.height}px`;
    
    document.body.appendChild(sparkle);
    
    setTimeout(() => sparkle.remove(), 1000);
}

// Add sparkle animation to CSS dynamically
const style = document.createElement('style');
style.textContent = `
    @keyframes sparkleFloat {
        0% {
            transform: translateY(0) scale(0);
            opacity: 1;
        }
        50% {
            transform: translateY(-20px) scale(1);
            opacity: 1;
        }
        100% {
            transform: translateY(-40px) scale(0);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Easter egg: Konami code for extra magic
let konamiCode = [];
const konamiPattern = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];

document.addEventListener('keydown', (e) => {
    konamiCode.push(e.key);
    konamiCode = konamiCode.slice(-10);
    
    if (konamiCode.join(',') === konamiPattern.join(',')) {
        activateSuperMagic();
    }
});

function activateSuperMagic() {
    showToast('🌟 SUPER MAGIC MODE ACTIVATED! 🌟', 'success');
    document.body.style.animation = 'rainbow 3s linear infinite';
    
    const rainbowStyle = document.createElement('style');
    rainbowStyle.textContent = `
        @keyframes rainbow {
            0% { filter: hue-rotate(0deg); }
            100% { filter: hue-rotate(360deg); }
        }
    `;
    document.head.appendChild(rainbowStyle);
    
    setTimeout(() => {
        document.body.style.animation = '';
        rainbowStyle.remove();
    }, 5000);
}

// ===== Config Management Functions =====

// Show save config modal
document.getElementById('save-config-btn')?.addEventListener('click', () => {
    document.getElementById('save-config-modal').style.display = 'flex';
});

// Show load config modal
document.getElementById('load-config-btn')?.addEventListener('click', () => {
    document.getElementById('load-config-modal').style.display = 'flex';
    loadConfigsList();
});

// Show manage configs modal
document.getElementById('manage-configs-btn')?.addEventListener('click', () => {
    document.getElementById('manage-configs-modal').style.display = 'flex';
    loadManageConfigsList();
});

// Get current configuration from form
function getCurrentConfig() {
    // Get dataset config
    const sourceType = document.getElementById('dataset-source-type').value;
    let datasetSource = { source_type: sourceType };

    if (sourceType === 'huggingface') {
        datasetSource.repo_id = document.getElementById('hf-repo-id').value;
        datasetSource.split = document.getElementById('hf-split').value;
    } else if (sourceType === 'local_file') {
        datasetSource.file_path = document.getElementById('local-file-path').value;
        datasetSource.file_format = document.getElementById('local-file-format').value;
    } else if (sourceType === 'upload') {
        datasetSource.dataset_id = window.uploadedDatasetId || '';
    }

    const formatType = document.getElementById('format-type').value;
    let datasetFormat = { format_type: formatType };

    if (formatType === 'qwen3') {
        datasetFormat.enable_thinking = document.getElementById('enable-thinking')?.checked ?? true;
    }

    // Get LoRA config
    const useLora = document.getElementById('use-lora').checked;
    let loraConfig = {};
    if (useLora) {
        loraConfig = {
            lora_r: parseInt(document.getElementById('lora-r').value),
            lora_alpha: parseInt(document.getElementById('lora-alpha').value),
            lora_dropout: parseFloat(document.getElementById('lora-dropout').value)
        };
    }

    // Build complete config
    const config = {
        base_model: document.getElementById('base-model').value,
        output_name: document.getElementById('output-name').value,
        use_lora: useLora,
        ...loraConfig,
        use_4bit: document.getElementById('use-4bit').checked,
        max_seq_length: parseInt(document.getElementById('max-seq-length').value),
        num_train_epochs: parseInt(document.getElementById('epochs').value),
        per_device_train_batch_size: parseInt(document.getElementById('batch-size').value),
        gradient_accumulation_steps: parseInt(document.getElementById('grad-accum').value),
        learning_rate: parseFloat(document.getElementById('learning-rate').value),
        warmup_ratio: parseFloat(document.getElementById('warmup-ratio').value),
        beta: parseFloat(document.getElementById('beta').value),
        dataset: {
            source: datasetSource,
            format: datasetFormat,
            test_size: parseFloat(document.getElementById('test-size').value)
        }
    };

    // Optional fields
    if (document.getElementById('hf-token').value) {
        config.hf_token = document.getElementById('hf-token').value;
    }
    if (document.getElementById('wandb-token').value) {
        config.wandb_api_key = document.getElementById('wandb-token').value;
        config.wandb_project = document.getElementById('wandb-project').value;
    }
    if (document.getElementById('push-to-hub').checked) {
        config.push_to_hub = true;
        config.hf_hub_repo_id = document.getElementById('hub-repo-id').value;
        config.hf_hub_private = document.getElementById('hub-private').checked;
    }

    return config;
}

// Save current configuration
async function saveCurrentConfig() {
    const name = document.getElementById('config-name').value.trim();
    if (!name) {
        showToast('Please enter a configuration name', 'error');
        return;
    }

    const description = document.getElementById('config-description').value.trim();
    const tagsInput = document.getElementById('config-tags').value.trim();
    const tags = tagsInput ? tagsInput.split(',').map(t => t.trim()).filter(t => t) : [];

    const config = getCurrentConfig();

    try {
        const response = await fetch(`${API_URL}/configs/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, config, description, tags })
        });

        const result = await response.json();

        if (response.ok) {
            showToast(`Configuration '${name}' saved successfully!`, 'success');
            document.getElementById('save-config-modal').style.display = 'none';
            // Clear form
            document.getElementById('config-name').value = '';
            document.getElementById('config-description').value = '';
            document.getElementById('config-tags').value = '';
        } else {
            showToast(`Failed to save: ${result.detail}`, 'error');
        }
    } catch (error) {
        showToast('Error saving configuration', 'error');
        console.error(error);
    }
}

// Load list of configs for loading
async function loadConfigsList() {
    const listEl = document.getElementById('config-list');
    listEl.innerHTML = '<p style="text-align: center; color: #888;">Loading...</p>';

    try {
        const response = await fetch(`${API_URL}/configs/list`);
        const result = await response.json();

        if (!response.ok) {
            listEl.innerHTML = '<p style="text-align: center; color: #ef4444;">Failed to load configurations</p>';
            return;
        }

        const configs = result.configs;

        if (configs.length === 0) {
            listEl.innerHTML = '<p style="text-align: center; color: #888;">No saved configurations found.</p>';
            return;
        }

        listEl.innerHTML = configs.map(cfg => `
            <div class="config-item" onclick="loadConfigByName('${cfg.filename}')">
                <h4>${cfg.name}</h4>
                ${cfg.description ? `<p>${cfg.description}</p>` : ''}
                <div class="config-meta">
                    ${cfg.tags.map(tag => `<span class="config-tag">${tag}</span>`).join('')}
                    <span class="config-date">Modified: ${new Date(cfg.modified_at).toLocaleString()}</span>
                </div>
            </div>
        `).join('');
    } catch (error) {
        listEl.innerHTML = '<p style="text-align: center; color: #ef4444;">Error loading configurations</p>';
        console.error(error);
    }
}

// Load a specific config and populate form
async function loadConfigByName(name) {
    try {
        const response = await fetch(`${API_URL}/configs/${name}`);
        const result = await response.json();

        if (!response.ok) {
            showToast(`Failed to load configuration: ${result.detail}`, 'error');
            return;
        }

        const config = result.config;

        // Populate form fields
        document.getElementById('base-model').value = config.base_model || '';
        document.getElementById('output-name').value = config.output_name || '';
        document.getElementById('use-lora').checked = config.use_lora ?? true;
        document.getElementById('use-4bit').checked = config.use_4bit ?? true;

        if (config.lora_r) document.getElementById('lora-r').value = config.lora_r;
        if (config.lora_alpha) document.getElementById('lora-alpha').value = config.lora_alpha;
        if (config.lora_dropout) document.getElementById('lora-dropout').value = config.lora_dropout;

        document.getElementById('max-seq-length').value = config.max_seq_length || 2048;
        document.getElementById('epochs').value = config.num_train_epochs || 1;
        document.getElementById('batch-size').value = config.per_device_train_batch_size || 1;
        document.getElementById('grad-accum').value = config.gradient_accumulation_steps || 4;
        document.getElementById('learning-rate').value = config.learning_rate || 0.000005;
        document.getElementById('warmup-ratio').value = config.warmup_ratio || 0.1;
        document.getElementById('beta').value = config.beta || 0.1;

        // Dataset config
        if (config.dataset) {
            if (config.dataset.source) {
                const source = config.dataset.source;
                document.getElementById('dataset-source-type').value = source.source_type || 'huggingface';

                if (source.repo_id) document.getElementById('hf-repo-id').value = source.repo_id;
                if (source.split) document.getElementById('hf-split').value = source.split;
                if (source.file_path) document.getElementById('local-file-path').value = source.file_path;
                if (source.file_format) document.getElementById('local-file-format').value = source.file_format;
            }

            if (config.dataset.format) {
                document.getElementById('format-type').value = config.dataset.format.format_type || 'chatml';
            }

            if (config.dataset.test_size) {
                document.getElementById('test-size').value = config.dataset.test_size;
            }
        }

        // Optional fields
        if (config.hf_token) document.getElementById('hf-token').value = config.hf_token;
        if (config.wandb_api_key) {
            document.getElementById('wandb-token').value = config.wandb_api_key;
            document.getElementById('wandb-project').value = config.wandb_project || '';
        }
        if (config.push_to_hub) {
            document.getElementById('push-to-hub').checked = true;
            document.getElementById('hub-repo-id').value = config.hf_hub_repo_id || '';
            document.getElementById('hub-private').checked = config.hf_hub_private ?? true;
        }

        showToast(`Configuration '${name}' loaded successfully!`, 'success');
        document.getElementById('load-config-modal').style.display = 'none';
    } catch (error) {
        showToast('Error loading configuration', 'error');
        console.error(error);
    }
}

// Load list of configs for management
async function loadManageConfigsList() {
    const listEl = document.getElementById('manage-config-list');
    listEl.innerHTML = '<p style="text-align: center; color: #888;">Loading...</p>';

    try {
        const response = await fetch(`${API_URL}/configs/list`);
        const result = await response.json();

        if (!response.ok) {
            listEl.innerHTML = '<p style="text-align: center; color: #ef4444;">Failed to load configurations</p>';
            return;
        }

        const configs = result.configs;

        if (configs.length === 0) {
            listEl.innerHTML = '<p style="text-align: center; color: #888;">No saved configurations found.</p>';
            return;
        }

        listEl.innerHTML = configs.map(cfg => `
            <div class="config-item">
                <h4>${cfg.name}</h4>
                ${cfg.description ? `<p>${cfg.description}</p>` : ''}
                <div class="config-meta">
                    ${cfg.tags.map(tag => `<span class="config-tag">${tag}</span>`).join('')}
                    <span class="config-date">Modified: ${new Date(cfg.modified_at).toLocaleString()}</span>
                </div>
                <div class="config-actions">
                    <button class="config-load-btn" onclick="loadConfigByName('${cfg.filename}'); document.getElementById('manage-configs-modal').style.display='none';">
                        📂 Load
                    </button>
                    <button class="config-delete-btn" onclick="deleteConfig('${cfg.filename}')">
                        🗑️ Delete
                    </button>
                </div>
            </div>
        `).join('');
    } catch (error) {
        listEl.innerHTML = '<p style="text-align: center; color: #ef4444;">Error loading configurations</p>';
        console.error(error);
    }
}

// Delete a configuration
async function deleteConfig(name) {
    if (!confirm(`Are you sure you want to delete the configuration '${name}'?`)) {
        return;
    }

    try {
        const response = await fetch(`${API_URL}/configs/${name}`, {
            method: 'DELETE'
        });

        const result = await response.json();

        if (response.ok) {
            showToast(`Configuration '${name}' deleted successfully!`, 'success');
            loadManageConfigsList(); // Reload the list
        } else {
            showToast(`Failed to delete: ${result.detail}`, 'error');
        }
    } catch (error) {
        showToast('Error deleting configuration', 'error');
        console.error(error);
    }
}

// ===== GPU Management Functions =====

// Refresh and display GPU list
async function handleRefreshGPUs() {
    const container = document.getElementById('gpu-list-container');
    const gpuSelect = document.getElementById('gpu-selection');
    const refreshButton = document.getElementById('refresh-gpu-button');

    // Show loading state
    refreshButton.disabled = true;
    refreshButton.textContent = '🔄 Loading...';
    container.innerHTML = '<div style="text-align: center; padding: 20px; color: #888;">Loading GPU information...</div>';

    try {
        const response = await fetch(`${API_URL}/gpu/list`);
        const data = await response.json();

        if (data.status === 'no_cuda') {
            container.innerHTML = `
                <div style="background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 10px;">
                    <div style="font-weight: bold; color: #856404; margin-bottom: 5px;">⚠️ No CUDA Available</div>
                    <div style="font-size: 0.9em; color: #666;">${data.message}</div>
                </div>
            `;
            gpuSelect.innerHTML = '<option value="">No GPUs available</option>';
            gpuSelect.disabled = true;
        } else if (data.gpus && data.gpus.length > 0) {
            // Display GPU cards
            container.innerHTML = data.gpus.map(gpu => `
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; margin-bottom: 10px; color: white;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <div style="font-weight: bold; font-size: 1.1em;">GPU ${gpu.index}: ${gpu.name}</div>
                        <div style="background: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 5px; font-size: 0.85em;">
                            ${gpu.compute_capability || 'N/A'}
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; font-size: 0.9em;">
                        <div>
                            <div style="opacity: 0.8; font-size: 0.85em;">Memory</div>
                            <div style="font-weight: bold;">${gpu.used_memory_mb} / ${gpu.total_memory_mb} MB</div>
                            <div style="font-size: 0.85em;">${gpu.memory_utilization_percent}% used</div>
                        </div>
                        ${gpu.gpu_utilization_percent !== null ? `
                        <div>
                            <div style="opacity: 0.8; font-size: 0.85em;">GPU Util</div>
                            <div style="font-weight: bold;">${gpu.gpu_utilization_percent}%</div>
                        </div>
                        ` : ''}
                        ${gpu.temperature_c !== null ? `
                        <div>
                            <div style="opacity: 0.8; font-size: 0.85em;">Temperature</div>
                            <div style="font-weight: bold;">${gpu.temperature_c}°C</div>
                        </div>
                        ` : ''}
                        ${gpu.power_usage_w !== null ? `
                        <div>
                            <div style="opacity: 0.8; font-size: 0.85em;">Power</div>
                            <div style="font-weight: bold;">${gpu.power_usage_w}W</div>
                        </div>
                        ` : ''}
                    </div>
                </div>
            `).join('');

            // Update select dropdown
            gpuSelect.innerHTML = '<option value="">Auto (use all available)</option>' +
                data.gpus.map(gpu => `<option value="${gpu.index}">GPU ${gpu.index}: ${gpu.name} (${gpu.free_memory_mb} MB free)</option>`).join('');
            gpuSelect.disabled = false;

            showToast(`Found ${data.gpus.length} GPU(s)`, 'success');
        } else {
            container.innerHTML = `
                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; color: #666;">
                    No GPUs detected
                </div>
            `;
            gpuSelect.innerHTML = '<option value="">No GPUs available</option>';
            gpuSelect.disabled = true;
        }
    } catch (error) {
        container.innerHTML = `
            <div style="background: #fee; border: 1px solid #fcc; padding: 15px; border-radius: 10px;">
                <div style="font-weight: bold; color: #c33; margin-bottom: 5px;">❌ Error Loading GPUs</div>
                <div style="font-size: 0.9em; color: #666;">${error.message}</div>
            </div>
        `;
        showToast('Failed to load GPU information', 'error');
        console.error(error);
    } finally {
        refreshButton.disabled = false;
        refreshButton.textContent = '🔄 Refresh GPU List';
    }
}

// Get selected GPU IDs for training config
function getSelectedGPUs() {
    const gpuSelect = document.getElementById('gpu-selection');
    const selected = Array.from(gpuSelect.selectedOptions)
        .map(opt => opt.value)
        .filter(val => val !== ''); // Filter out the "Auto" option

    // If nothing selected or "Auto" is selected, return null (use all GPUs)
    return selected.length > 0 ? selected.map(Number) : null;
}

// Refresh GPU stats while monitoring a job
async function refreshJobGPUs() {
    const container = document.getElementById('gpu-monitor-cards');

    if (!container) return;

    container.innerHTML = '<div style="text-align: center; padding: 20px; color: #888;">Loading GPU stats...</div>';

    try {
        const response = await fetch(`${API_URL}/gpu/list`);
        const data = await response.json();

        if (data.status === 'success' && data.gpus && data.gpus.length > 0) {
            container.innerHTML = data.gpus.map(gpu => `
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 12px; border-radius: 8px; margin-bottom: 8px; color: white; font-size: 0.9em;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div style="font-weight: bold;">GPU ${gpu.index}: ${gpu.name}</div>
                        <div style="background: rgba(255,255,255,0.2); padding: 3px 8px; border-radius: 4px; font-size: 0.85em;">
                            ${gpu.memory_utilization_percent}% mem
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 8px; font-size: 0.85em;">
                        <div>
                            <div style="opacity: 0.8;">Memory</div>
                            <div style="font-weight: bold;">${gpu.used_memory_mb}/${gpu.total_memory_mb} MB</div>
                        </div>
                        ${gpu.gpu_utilization_percent !== null ? `
                        <div>
                            <div style="opacity: 0.8;">GPU Util</div>
                            <div style="font-weight: bold;">${gpu.gpu_utilization_percent}%</div>
                        </div>
                        ` : ''}
                        ${gpu.temperature_c !== null ? `
                        <div>
                            <div style="opacity: 0.8;">Temp</div>
                            <div style="font-weight: bold;">${gpu.temperature_c}°C</div>
                        </div>
                        ` : ''}
                    </div>
                </div>
            `).join('');

            // Show the section if hidden
            document.getElementById('gpu-monitor-section').style.display = 'block';
        } else {
            container.innerHTML = '<div style="text-align: center; padding: 15px; color: #888;">No GPU data available</div>';
        }
    } catch (error) {
        container.innerHTML = `<div style="background: #fee; padding: 10px; border-radius: 8px; color: #c33; text-align: center;">Error: ${error.message}</div>`;
        console.error('Failed to refresh GPU stats:', error);
    }
}