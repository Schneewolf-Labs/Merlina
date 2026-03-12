// Unit tests for theme.js - ThemeManager logic
// Run with: node --test tests/frontend/test_theme.mjs

import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

// ─── DOM & localStorage shims ───────────────────────────────────────────────

let storedTheme = null;
let appliedTheme = null;
let createdElements = [];
let matchMediaDark = false;
let mediaListeners = [];

function resetShims() {
    storedTheme = null;
    appliedTheme = null;
    createdElements = [];
    matchMediaDark = false;
    mediaListeners = [];
}

globalThis.localStorage = {
    getItem(key) {
        if (key === 'merlina-theme') return storedTheme;
        return null;
    },
    setItem(key, val) {
        if (key === 'merlina-theme') storedTheme = val;
    },
    removeItem(key) {
        if (key === 'merlina-theme') storedTheme = null;
    },
};

globalThis.document = {
    documentElement: {
        setAttribute(name, val) {
            if (name === 'data-theme') appliedTheme = val;
        },
        getAttribute(name) {
            if (name === 'data-theme') return appliedTheme;
            return null;
        },
    },
    createElement(tag) {
        const el = {
            _tag: tag,
            _attrs: {},
            _listeners: {},
            _classes: new Set(),
            id: '',
            innerHTML: '',
            textContent: '',
            style: {},
            setAttribute(n, v) { this._attrs[n] = v; },
            getAttribute(n) { return this._attrs[n] ?? null; },
            addEventListener(evt, fn) {
                if (!this._listeners[evt]) this._listeners[evt] = [];
                this._listeners[evt].push(fn);
            },
            get className() { return Array.from(this._classes).join(' '); },
            set className(val) {
                this._classes.clear();
                val.split(' ').filter(Boolean).forEach(c => this._classes.add(c));
            },
            get classList() {
                const self = this;
                return {
                    add(c) { self._classes.add(c); },
                    remove(c) { self._classes.delete(c); },
                };
            },
            appendChild(child) { createdElements.push(child); },
        };
        return el;
    },
    getElementById(id) {
        if (id === 'theme-toggle') {
            return createdElements.find(el => el.id === 'theme-toggle') || null;
        }
        return null;
    },
    querySelector(sel) {
        if (sel === '.logo') {
            return {
                appendChild(child) { createdElements.push(child); },
            };
        }
        return null;
    },
    querySelectorAll() { return []; },
    addEventListener() {},
    body: { style: {} },
    head: { appendChild() {} },
};

globalThis.window = {
    location: { protocol: 'https:', host: 'localhost:8000' },
    addEventListener() {},
    removeEventListener() {},
    confirm: () => true,
    getComputedStyle: () => ({ display: 'none' }),
    matchMedia(query) {
        return {
            matches: matchMediaDark,
            addEventListener(evt, fn) {
                mediaListeners.push(fn);
            },
        };
    },
};

globalThis.console = { ...console };
globalThis.WebSocket = class WebSocket {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSING = 2;
    static CLOSED = 3;
};
globalThis.AbortController = AbortController;
globalThis.FormData = class FormData { append() {} };
globalThis.XMLHttpRequest = class XMLHttpRequest {
    open() {} send() {} upload = { addEventListener() {} };
    addEventListener() {}
};

// ─── Import after shims ─────────────────────────────────────────────────────

const { ThemeManager } = await import('../../frontend/js/theme.js');

// ═════════════════════════════════════════════════════════════════════════════
// ThemeManager tests
// ═════════════════════════════════════════════════════════════════════════════

describe('ThemeManager', () => {
    beforeEach(() => {
        resetShims();
    });

    describe('constructor and initialization', () => {
        it('defaults to light when no stored preference and system is light', () => {
            matchMediaDark = false;
            const tm = new ThemeManager();
            assert.equal(tm.theme, 'light');
        });

        it('uses system dark preference when no stored theme', () => {
            matchMediaDark = true;
            const tm = new ThemeManager();
            assert.equal(tm.theme, 'dark');
        });

        it('uses stored theme over system preference', () => {
            storedTheme = 'dark';
            matchMediaDark = false;
            const tm = new ThemeManager();
            assert.equal(tm.theme, 'dark');
        });

        it('applies theme to document on init', () => {
            storedTheme = 'dark';
            const tm = new ThemeManager();
            assert.equal(appliedTheme, 'dark');
        });
    });

    describe('getStoredTheme', () => {
        it('returns null when nothing stored', () => {
            const tm = new ThemeManager();
            storedTheme = null;
            assert.equal(tm.getStoredTheme(), null);
        });

        it('returns stored theme value', () => {
            storedTheme = 'dark';
            const tm = new ThemeManager();
            assert.equal(tm.getStoredTheme(), 'dark');
        });
    });

    describe('getSystemPreference', () => {
        it('returns dark when system prefers dark', () => {
            matchMediaDark = true;
            const tm = new ThemeManager();
            assert.equal(tm.getSystemPreference(), 'dark');
        });

        it('returns light when system prefers light', () => {
            matchMediaDark = false;
            const tm = new ThemeManager();
            assert.equal(tm.getSystemPreference(), 'light');
        });
    });

    describe('setTheme', () => {
        it('applies theme and persists by default', () => {
            const tm = new ThemeManager();
            tm.setTheme('dark');
            assert.equal(tm.theme, 'dark');
            assert.equal(appliedTheme, 'dark');
            assert.equal(storedTheme, 'dark');
        });

        it('applies theme without persisting when persist=false', () => {
            storedTheme = null;
            const tm = new ThemeManager();
            tm.setTheme('dark', false);
            assert.equal(tm.theme, 'dark');
            assert.equal(appliedTheme, 'dark');
            assert.equal(storedTheme, null); // Not persisted
        });
    });

    describe('toggle', () => {
        it('toggles from light to dark', () => {
            matchMediaDark = false;
            const tm = new ThemeManager();
            assert.equal(tm.theme, 'light');
            tm.toggle();
            assert.equal(tm.theme, 'dark');
        });

        it('toggles from dark to light', () => {
            storedTheme = 'dark';
            const tm = new ThemeManager();
            assert.equal(tm.theme, 'dark');
            tm.toggle();
            assert.equal(tm.theme, 'light');
        });

        it('persists toggled theme', () => {
            matchMediaDark = false;
            const tm = new ThemeManager();
            tm.toggle();
            assert.equal(storedTheme, 'dark');
        });
    });

    describe('getToggleIcon', () => {
        it('returns sun icon when theme is dark (to switch to light)', () => {
            storedTheme = 'dark';
            const tm = new ThemeManager();
            const icon = tm.getToggleIcon();
            assert.ok(icon.includes('circle'), 'Sun icon should contain circle element');
        });

        it('returns moon icon when theme is light (to switch to dark)', () => {
            matchMediaDark = false;
            const tm = new ThemeManager();
            const icon = tm.getToggleIcon();
            assert.ok(icon.includes('path'), 'Moon icon should contain path element');
        });
    });

    describe('createToggleButton', () => {
        it('creates a button element with correct attributes', () => {
            matchMediaDark = false;
            const tm = new ThemeManager();
            // Button should have been created and appended
            const btn = createdElements.find(el => el.id === 'theme-toggle');
            assert.ok(btn, 'Toggle button should be created');
            // className is set as string, not via classList
            assert.ok(btn._classes.has('theme-toggle-btn'), 'Button should have theme-toggle-btn class');
            // Initial label is set by createToggleButton, then updated by updateToggleButton
            assert.ok(btn._attrs['aria-label'], 'Button should have aria-label');
        });
    });
});
