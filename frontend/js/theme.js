// Theme Module - Dark/Light mode handling

/**
 * ThemeManager handles theme switching and persistence
 */
class ThemeManager {
    constructor() {
        this.storageKey = 'merlina-theme';
        this.theme = this.getStoredTheme() || this.getSystemPreference();
        this.init();
    }

    /**
     * Initialize theme on page load
     */
    init() {
        // Apply theme immediately to prevent flash
        this.applyTheme(this.theme);

        // Listen for system preference changes
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
                // Only auto-switch if user hasn't set a preference
                if (!this.getStoredTheme()) {
                    this.setTheme(e.matches ? 'dark' : 'light', false);
                }
            });
        }

        // Create toggle button
        this.createToggleButton();
    }

    /**
     * Get stored theme preference
     */
    getStoredTheme() {
        try {
            return localStorage.getItem(this.storageKey);
        } catch {
            return null;
        }
    }

    /**
     * Get system color scheme preference
     */
    getSystemPreference() {
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            return 'dark';
        }
        return 'light';
    }

    /**
     * Apply theme to document
     */
    applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        this.theme = theme;
        this.updateToggleButton();
    }

    /**
     * Set theme and optionally persist
     */
    setTheme(theme, persist = true) {
        this.applyTheme(theme);
        if (persist) {
            try {
                localStorage.setItem(this.storageKey, theme);
            } catch {
                // localStorage not available
            }
        }
    }

    /**
     * Toggle between light and dark themes
     */
    toggle() {
        const newTheme = this.theme === 'dark' ? 'light' : 'dark';
        this.setTheme(newTheme);
    }

    /**
     * Create the theme toggle button in the header
     */
    createToggleButton() {
        const logo = document.querySelector('.logo');
        if (!logo) return;

        const button = document.createElement('button');
        button.id = 'theme-toggle';
        button.className = 'theme-toggle-btn';
        button.setAttribute('aria-label', 'Toggle dark mode');
        button.setAttribute('title', 'Toggle dark/light mode');
        button.innerHTML = this.getToggleIcon();

        button.addEventListener('click', () => {
            this.toggle();
            // Add a subtle animation
            button.classList.add('theme-toggle-animate');
            setTimeout(() => button.classList.remove('theme-toggle-animate'), 300);
        });

        // Insert after tagline
        logo.appendChild(button);
    }

    /**
     * Update toggle button icon
     */
    updateToggleButton() {
        const button = document.getElementById('theme-toggle');
        if (button) {
            button.innerHTML = this.getToggleIcon();
            button.setAttribute('aria-label',
                this.theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'
            );
        }
    }

    /**
     * Get the appropriate icon for current theme
     */
    getToggleIcon() {
        if (this.theme === 'dark') {
            // Sun icon for switching to light
            return `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="5"/>
                <line x1="12" y1="1" x2="12" y2="3"/>
                <line x1="12" y1="21" x2="12" y2="23"/>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                <line x1="1" y1="12" x2="3" y2="12"/>
                <line x1="21" y1="12" x2="23" y2="12"/>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
            </svg>`;
        } else {
            // Moon icon for switching to dark
            return `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
            </svg>`;
        }
    }
}

// Export for use in app.js
export { ThemeManager };
