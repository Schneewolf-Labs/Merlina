/**
 * Layout Manager - Handles layout state, panel visibility, and responsiveness
 * Merlina Modular Frontend v2.0
 */

export class LayoutManager {
    constructor() {
        this.sidebar = document.getElementById('sidebar');
        this.rightPanel = document.getElementById('right-panel');
        this.backdrop = document.getElementById('backdrop');
        this.sidebarCollapsed = false;
        this.panelVisible = true;
        this.currentView = 'dashboard';

        this.init();
    }

    /**
     * Initialize layout manager
     */
    init() {
        this.setupSidebarToggle();
        this.setupPanelToggle();
        this.setupBackdrop();
        this.setupResponsive();
        this.loadLayoutState();
    }

    /**
     * Setup sidebar toggle functionality
     */
    setupSidebarToggle() {
        // Mobile sidebar toggle
        const hamburger = document.getElementById('sidebar-toggle');
        if (hamburger) {
            hamburger.addEventListener('click', () => this.toggleSidebar());
        }

        // Handle navigation item clicks
        const navItems = document.querySelectorAll('.sidebar-nav-item');
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const view = item.dataset.view;
                if (view) {
                    this.setActiveNavItem(item);
                    this.navigateToView(view);

                    // Close sidebar on mobile after navigation
                    if (window.innerWidth < 640) {
                        this.toggleSidebar(false);
                    }
                }
            });
        });
    }

    /**
     * Setup right panel toggle
     */
    setupPanelToggle() {
        const toggleBtn = document.getElementById('toggle-panel');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.togglePanel());
        }
    }

    /**
     * Setup backdrop click handler
     */
    setupBackdrop() {
        this.backdrop?.addEventListener('click', () => {
            this.toggleSidebar(false);
            this.togglePanel(false);
        });
    }

    /**
     * Setup responsive behavior
     */
    setupResponsive() {
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.handleResize();
            }, 150);
        });

        // Initial check
        this.handleResize();
    }

    /**
     * Handle window resize
     */
    handleResize() {
        const width = window.innerWidth;

        // Auto-hide panel on tablet and below
        if (width < 1280) {
            this.togglePanel(false);
        } else {
            // Show panel on desktop if it was hidden
            if (!this.panelVisible) {
                this.togglePanel(true);
            }
        }

        // Auto-collapse sidebar on mobile
        if (width < 768) {
            this.sidebar?.classList.add('collapsed');
            this.sidebarCollapsed = true;
        } else if (width >= 768 && this.sidebarCollapsed) {
            this.sidebar?.classList.remove('collapsed');
            this.sidebarCollapsed = false;
        }
    }

    /**
     * Toggle sidebar visibility
     * @param {boolean} force - Force open/close
     */
    toggleSidebar(force) {
        const show = force !== undefined ? force : !this.sidebar?.classList.contains('visible');

        if (show) {
            this.sidebar?.classList.add('visible');
            this.backdrop?.classList.add('visible');
        } else {
            this.sidebar?.classList.remove('visible');
            this.backdrop?.classList.remove('visible');
        }

        this.saveLayoutState();
    }

    /**
     * Toggle right panel visibility
     * @param {boolean} force - Force open/close
     */
    togglePanel(force) {
        this.panelVisible = force !== undefined ? force : !this.panelVisible;

        if (this.panelVisible) {
            this.rightPanel?.classList.add('visible');
            this.rightPanel?.classList.remove('hidden');
            if (window.innerWidth < 1280) {
                this.backdrop?.classList.add('visible');
            }
        } else {
            this.rightPanel?.classList.remove('visible');
            this.rightPanel?.classList.add('hidden');
            if (window.innerWidth < 1280) {
                this.backdrop?.classList.remove('visible');
            }
        }

        this.saveLayoutState();
    }

    /**
     * Set active navigation item
     * @param {HTMLElement} activeItem - The nav item to set as active
     */
    setActiveNavItem(activeItem) {
        // Remove active class from all items
        document.querySelectorAll('.sidebar-nav-item').forEach(item => {
            item.classList.remove('active');
        });

        // Add active class to clicked item
        activeItem.classList.add('active');
    }

    /**
     * Navigate to a view
     * @param {string} viewName - Name of the view to navigate to
     */
    navigateToView(viewName) {
        this.currentView = viewName;

        // Emit navigation event
        window.dispatchEvent(new CustomEvent('navigate', {
            detail: { view: viewName }
        }));

        this.saveLayoutState();
    }

    /**
     * Get current view
     * @returns {string} Current view name
     */
    getCurrentView() {
        return this.currentView;
    }

    /**
     * Save layout state to localStorage
     */
    saveLayoutState() {
        try {
            const state = {
                sidebarCollapsed: this.sidebarCollapsed,
                panelVisible: this.panelVisible,
                currentView: this.currentView
            };
            localStorage.setItem('merlina_layout_state', JSON.stringify(state));
        } catch (error) {
            console.error('Failed to save layout state:', error);
        }
    }

    /**
     * Load layout state from localStorage
     */
    loadLayoutState() {
        try {
            const saved = localStorage.getItem('merlina_layout_state');
            if (saved) {
                const state = JSON.parse(saved);
                this.currentView = state.currentView || 'dashboard';

                // Don't restore panel state on mobile
                if (window.innerWidth >= 1280) {
                    this.panelVisible = state.panelVisible !== false;
                }
            }
        } catch (error) {
            console.error('Failed to load layout state:', error);
        }
    }

    /**
     * Update active jobs badge count
     * @param {number} count - Number of active jobs
     */
    updateActiveJobsCount(count) {
        const badge = document.getElementById('active-jobs-count');
        if (badge) {
            badge.textContent = count;
            badge.style.display = count > 0 ? 'block' : 'none';
        }
    }
}
