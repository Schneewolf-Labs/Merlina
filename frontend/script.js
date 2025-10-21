// Merlina Frontend JavaScript

const API_URL = 'http://localhost:8000';
let activeJobs = {};
let currentJobId = null;
let pollInterval = null;
let uploadedDatasetId = null; // Store uploaded dataset ID

// DOM Elements
const form = document.getElementById('training-form');
const jobsSection = document.getElementById('jobs-section');
const jobsContainer = document.getElementById('jobs-container');
const modal = document.getElementById('job-modal');
const closeModal = document.querySelector('.close');
const toast = document.getElementById('toast');

// Form Elements
const pushHub = document.getElementById('push-hub');
const hfTokenGroup = document.getElementById('hf-token-group');
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

    // Show/hide HF token field based on push to hub
    pushHub.addEventListener('change', (e) => {
        hfTokenGroup.style.display = e.target.checked ? 'block' : 'none';
    });

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
    datasetFormatType.addEventListener('change', (e) => {
        customFormatConfig.style.display = e.target.value === 'custom' ? 'block' : 'none';
    });

    // Upload dataset button
    uploadButton.addEventListener('click', handleDatasetUpload);

    // Preview dataset buttons
    previewButton.addEventListener('click', handleDatasetPreview);
    previewFormattedButton.addEventListener('click', handleFormattedPreview);

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
            'chatml': 'ChatML',
            'llama3': 'Llama 3',
            'mistral': 'Mistral Instruct',
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

    return config;
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

        // Advanced training settings
        warmup_ratio: parseFloat(document.getElementById('warmup-ratio').value),
        eval_steps: parseFloat(document.getElementById('eval-steps').value),

        // Options
        use_4bit: document.getElementById('use-4bit').checked,
        use_wandb: document.getElementById('use-wandb').checked,
        push_to_hub: document.getElementById('push-hub').checked,

        // API Keys
        wandb_key: document.getElementById('wandb-key').value || null,
        hf_token: document.getElementById('hf-token').value || null
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
        
        // Reset form
        form.reset();
        
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
    card.onclick = () => showJobDetails(jobId);
    
    const statusClass = `status-${job.status}`;
    const progressPercent = Math.round((job.progress || 0) * 100);
    
    card.innerHTML = `
        <div class="job-card-header">
            <h4>${activeJobs[jobId]?.name || jobId}</h4>
            <span class="job-status-badge ${statusClass}">${job.status}</span>
        </div>
        <div class="progress-bar">
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
        
        // Show W&B link if using wandb
        const wandbLink = document.getElementById('wandb-link');
        if (activeJobs[currentJobId]?.config?.use_wandb && status.status === 'training') {
            wandbLink.style.display = 'block';
            // In a real app, you'd get the actual W&B run URL from the backend
            wandbLink.href = `https://wandb.ai/your-team/merlin-training/runs/${currentJobId}`;
        }
        
        // Handle completion
        if (status.status === 'completed') {
            clearInterval(pollInterval);
            showToast('✅ Training completed successfully!', 'success');
            document.getElementById('stop-button').disabled = true;
        } else if (status.status === 'failed') {
            clearInterval(pollInterval);
            showToast(`❌ Training failed: ${status.error || 'Unknown error'}`, 'error');
            document.getElementById('stop-button').disabled = true;
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

// Stop training (placeholder - backend doesn't implement this yet)
async function stopTraining() {
    if (!currentJobId) return;
    
    if (confirm('Are you sure you want to stop this training?')) {
        try {
            // In a real implementation, you'd call an endpoint to stop the job
            showToast('⚠️ Stop functionality not implemented in backend yet', 'warning');
        } catch (error) {
            showToast(`❌ Failed to stop training: ${error.message}`, 'error');
        }
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