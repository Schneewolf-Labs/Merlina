/**
 * Merlina Modular Frontend v2.0
 * Main Application Entry Point
 */

import { LayoutManager } from './core/layout-manager.js';
import { TrainingModeManager } from './managers/training-mode-manager.js';
import { DashboardView } from './views/dashboard-view.js';

/**
 * Main Application Class
 */
class MerlinaAppV2 {
    constructor() {
        // Core managers
        this.layoutManager = null;
        this.trainingModeManager = null;

        // Views
        this.views = new Map();
        this.currentView = null;

        // State
        this.stats = {
            activeJobs: 0,
            completedJobs: 0,
            savedConfigs: 0,
            gpus: 'Loading...'
        };

        // Make available globally for debugging
        window.merlinaAppV2 = this;
    }

    /**
     * Initialize the application
     */
    async init() {
        console.log('🧙 Initializing Merlina v2.0...');

        try {
            // Initialize managers
            this.initManagers();

            // Register views
            this.registerViews();

            // Setup event listeners
            this.setupEventListeners();

            // Load initial view
            this.loadInitialView();

            // Start background tasks
            this.startBackgroundTasks();

            console.log('✨ Merlina v2.0 initialized successfully!');
        } catch (error) {
            console.error('Failed to initialize Merlina:', error);
            this.showError('Failed to initialize application. Please refresh the page.');
        }
    }

    /**
     * Initialize core managers
     */
    initManagers() {
        // Layout manager
        this.layoutManager = new LayoutManager();

        // Training mode manager
        this.trainingModeManager = new TrainingModeManager();
        this.trainingModeManager.loadMode();

        // Make managers globally accessible
        window.layoutManager = this.layoutManager;
        window.trainingModeManager = this.trainingModeManager;
    }

    /**
     * Register all views
     */
    registerViews() {
        // Dashboard view
        this.views.set('dashboard', new DashboardView(this.trainingModeManager));

        // Placeholder views (to be implemented)
        this.views.set('model', { render: () => '<div class="card"><h2>Model Selection</h2><p>Coming soon...</p></div>', attachEventListeners: () => {} });
        this.views.set('dataset', { render: () => '<div class="card"><h2>Dataset Manager</h2><p>Coming soon...</p></div>', attachEventListeners: () => {} });
        this.views.set('training', { render: () => '<div class="card"><h2>Training Config</h2><p>Coming soon...</p></div>', attachEventListeners: () => {} });
        this.views.set('active-jobs', { render: () => '<div class="card"><h2>Active Jobs</h2><p>Coming soon...</p></div>', attachEventListeners: () => {} });
        this.views.set('job-history', { render: () => '<div class="card"><h2>Job History</h2><p>Coming soon...</p></div>', attachEventListeners: () => {} });
        this.views.set('configs', { render: () => '<div class="card"><h2>Saved Configs</h2><p>Coming soon...</p></div>', attachEventListeners: () => {} });
        this.views.set('gpu', { render: () => '<div class="card"><h2>GPU Manager</h2><p>Coming soon...</p></div>', attachEventListeners: () => {} });
        this.views.set('analytics', { render: () => '<div class="card"><h2>Analytics</h2><p>Coming soon...</p></div>', attachEventListeners: () => {} });
    }

    /**
     * Setup global event listeners
     */
    setupEventListeners() {
        // Navigation events
        window.addEventListener('navigate', (e) => {
            const { view } = e.detail;
            this.navigateToView(view);
        });

        // Training mode changes
        window.addEventListener('trainingModeChanged', (e) => {
            console.log('Training mode changed:', e.detail);
            this.handleModeChange(e.detail);
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcut(e);
        });
    }

    /**
     * Load initial view
     */
    loadInitialView() {
        const initialView = this.layoutManager.getCurrentView() || 'dashboard';
        this.navigateToView(initialView);
    }

    /**
     * Navigate to a view
     * @param {string} viewName - Name of view to navigate to
     */
    navigateToView(viewName) {
        const view = this.views.get(viewName);

        if (!view) {
            console.error(`View not found: ${viewName}`);
            this.showToast('View not found', 'danger');
            return;
        }

        console.log(`Navigating to: ${viewName}`);

        // Update layout manager
        this.layoutManager.currentView = viewName;

        // Render view
        const container = document.getElementById('view-container');
        if (container) {
            container.innerHTML = view.render();

            // Attach view-specific event listeners
            if (view.attachEventListeners) {
                view.attachEventListeners();
            }

            // Scroll to top
            container.scrollTop = 0;
        }

        this.currentView = viewName;

        // Update stats if dashboard
        if (viewName === 'dashboard') {
            this.updateDashboardStats();
        }
    }

    /**
     * Handle training mode change
     * @param {object} detail - Mode change details
     */
    handleModeChange(detail) {
        // Update UI elements that depend on mode
        const { mode, config } = detail;

        // Show toast notification
        this.showToast(
            `Switched to ${config.fullName} mode`,
            'success'
        );
    }

    /**
     * Handle keyboard shortcuts
     * @param {KeyboardEvent} e - Keyboard event
     */
    handleKeyboardShortcut(e) {
        // Ctrl/Cmd + K: Quick command palette (future)
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            console.log('Quick command palette (not implemented)');
        }

        // Ctrl/Cmd + /: Toggle sidebar
        if ((e.ctrlKey || e.metaKey) && e.key === '/') {
            e.preventDefault();
            this.layoutManager.toggleSidebar();
        }

        // Esc: Close overlays
        if (e.key === 'Escape') {
            this.layoutManager.togglePanel(false);
            this.layoutManager.toggleSidebar(false);
        }
    }

    /**
     * Start background tasks (polling, etc.)
     */
    startBackgroundTasks() {
        // Poll for stats every 5 seconds
        setInterval(() => {
            this.fetchStats();
        }, 5000);

        // Initial fetch
        this.fetchStats();
    }

    /**
     * Fetch application stats
     */
    async fetchStats() {
        try {
            // Fetch from API (placeholder for now)
            // const response = await fetch('/api/stats');
            // const stats = await response.json();

            // Mock data for now
            const stats = {
                activeJobs: 0,
                completedJobs: 0,
                savedConfigs: Object.keys(localStorage).filter(k => k.startsWith('merlina_config_')).length,
                gpus: '0/0'
            };

            this.updateStats(stats);
        } catch (error) {
            console.error('Failed to fetch stats:', error);
        }
    }

    /**
     * Update application stats
     * @param {object} stats - Stats object
     */
    updateStats(stats) {
        this.stats = { ...this.stats, ...stats };

        // Update dashboard if visible
        if (this.currentView === 'dashboard') {
            this.updateDashboardStats();
        }

        // Update active jobs count in sidebar
        if (stats.activeJobs !== undefined) {
            this.layoutManager.updateActiveJobsCount(stats.activeJobs);
        }
    }

    /**
     * Update dashboard stats
     */
    updateDashboardStats() {
        const view = this.views.get('dashboard');
        if (view && view.updateStats) {
            view.updateStats(this.stats);
        }
    }

    /**
     * Show toast notification
     * @param {string} message - Toast message
     * @param {string} type - Toast type (success, danger, warning, info)
     */
    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;

        const icons = {
            success: '✅',
            danger: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };

        toast.innerHTML = `
            <div class="toast-icon">${icons[type] || icons.info}</div>
            <div class="toast-content">
                <div class="toast-message">${message}</div>
            </div>
        `;

        container.appendChild(toast);

        // Auto-dismiss after 3 seconds
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    /**
     * Show error message
     * @param {string} message - Error message
     */
    showError(message) {
        this.showToast(message, 'danger');
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = new MerlinaAppV2();
    app.init();
});

export { MerlinaAppV2 };
