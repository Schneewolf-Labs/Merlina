// Unit tests for api.js - Error handling and categorization logic
// Run with: node --test tests/frontend/test_api.mjs

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

// ─── Browser API shims ──────────────────────────────────────────────────────

globalThis.window = {
    location: { protocol: 'https:', host: 'localhost:8000' },
    addEventListener: () => {},
};
globalThis.document = {
    createElement: () => ({}),
    getElementById: () => null,
    querySelectorAll: () => [],
    addEventListener: () => {},
};
globalThis.console = { ...console };
globalThis.WebSocket = class WebSocket {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSING = 2;
    static CLOSED = 3;
};

// Shim fetch and AbortController (already in Node 22 but need for module load)
globalThis.AbortController = AbortController;
globalThis.FormData = class FormData { append() {} };
globalThis.XMLHttpRequest = class XMLHttpRequest {
    open() {} send() {} upload = { addEventListener() {} };
    addEventListener() {}
};

const { APIError, ErrorType, WebSocketManager } = await import(
    '../../frontend/js/api.js'
);

// ═════════════════════════════════════════════════════════════════════════════
// APIError tests
// ═════════════════════════════════════════════════════════════════════════════

describe('APIError', () => {
    it('is an instance of Error', () => {
        const err = new APIError('test');
        assert.ok(err instanceof Error);
    });

    it('stores type and status code', () => {
        const err = new APIError('test', ErrorType.AUTH, 401, { key: 'val' });
        assert.equal(err.type, 'auth');
        assert.equal(err.statusCode, 401);
        assert.deepEqual(err.details, { key: 'val' });
    });

    it('defaults to UNKNOWN type', () => {
        const err = new APIError('test');
        assert.equal(err.type, ErrorType.UNKNOWN);
    });

    describe('getUserMessage', () => {
        it('returns network message for NETWORK type', () => {
            const err = new APIError('fail', ErrorType.NETWORK);
            const msg = err.getUserMessage();
            assert.match(msg, /network/i);
        });

        it('returns timeout message for TIMEOUT type', () => {
            const err = new APIError('fail', ErrorType.TIMEOUT);
            const msg = err.getUserMessage();
            assert.match(msg, /timed out/i);
        });

        it('returns server message for 500+ errors', () => {
            const err = new APIError('fail', ErrorType.SERVER, 502);
            const msg = err.getUserMessage();
            assert.match(msg, /server error/i);
        });

        it('returns original message for 4xx server errors', () => {
            const err = new APIError('Bad request payload', ErrorType.SERVER, 400);
            const msg = err.getUserMessage();
            assert.equal(msg, 'Bad request payload');
        });

        it('returns auth message for AUTH type', () => {
            const err = new APIError('fail', ErrorType.AUTH);
            const msg = err.getUserMessage();
            assert.match(msg, /authentication/i);
        });

        it('returns not found message for NOT_FOUND type', () => {
            const err = new APIError('fail', ErrorType.NOT_FOUND);
            const msg = err.getUserMessage();
            assert.match(msg, /not found/i);
        });

        it('returns validation message for VALIDATION type', () => {
            const err = new APIError('field X is wrong', ErrorType.VALIDATION);
            const msg = err.getUserMessage();
            assert.match(msg, /validation error/i);
            assert.match(msg, /field X is wrong/);
        });

        it('falls back to generic message for UNKNOWN type', () => {
            const err = new APIError('', ErrorType.UNKNOWN);
            const msg = err.getUserMessage();
            assert.match(msg, /unexpected error/i);
        });
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// ErrorType enum tests
// ═════════════════════════════════════════════════════════════════════════════

describe('ErrorType', () => {
    it('has all expected types', () => {
        const expectedTypes = ['NETWORK', 'TIMEOUT', 'SERVER', 'VALIDATION', 'AUTH', 'NOT_FOUND', 'UNKNOWN'];
        for (const type of expectedTypes) {
            assert.ok(ErrorType[type] !== undefined, `Missing error type: ${type}`);
        }
    });

    it('has unique values', () => {
        const values = Object.values(ErrorType);
        const unique = new Set(values);
        assert.equal(values.length, unique.size, 'Error types should have unique values');
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// WebSocketManager tests (stateful, no real connection)
// ═════════════════════════════════════════════════════════════════════════════

describe('WebSocketManager', () => {
    it('initializes with default state', () => {
        const ws = new WebSocketManager();
        assert.equal(ws.jobId, null);
        assert.equal(ws.reconnectAttempts, 0);
        assert.equal(ws.maxReconnectAttempts, 10);
        assert.equal(ws.isIntentionalClose, false);
    });

    it('calculates exponential backoff delay', () => {
        const ws = new WebSocketManager();

        ws.reconnectAttempts = 0;
        const delay0 = ws.getReconnectDelay();
        assert.ok(delay0 >= 1000 && delay0 <= 1300, `Expected ~1000ms, got ${delay0}`);

        ws.reconnectAttempts = 1;
        const delay1 = ws.getReconnectDelay();
        assert.ok(delay1 >= 2000 && delay1 <= 2600, `Expected ~2000ms, got ${delay1}`);

        ws.reconnectAttempts = 2;
        const delay2 = ws.getReconnectDelay();
        assert.ok(delay2 >= 4000 && delay2 <= 5200, `Expected ~4000ms, got ${delay2}`);
    });

    it('caps reconnect delay at maxReconnectDelay', () => {
        const ws = new WebSocketManager();
        ws.reconnectAttempts = 20; // Very high
        const delay = ws.getReconnectDelay();
        // With jitter, max should be maxReconnectDelay * 1.3
        assert.ok(delay <= ws.maxReconnectDelay * 1.3 + 1,
            `Delay ${delay} should be capped at ~${ws.maxReconnectDelay}`);
    });

    it('reports disconnected when no socket exists', () => {
        const ws = new WebSocketManager();
        assert.equal(ws.getStatus(), 'disconnected');
        assert.equal(ws.isConnected(), false);
    });

    it('disconnect cleans up all state', () => {
        const ws = new WebSocketManager();
        ws.jobId = 'test-123';
        ws.socket = { close: () => {} };
        ws.disconnect();
        assert.equal(ws.jobId, null);
        assert.equal(ws.socket, null);
        assert.equal(ws.isIntentionalClose, true);
    });

    it('disconnect clears jobId even without socket', () => {
        const ws = new WebSocketManager();
        ws.jobId = 'orphaned-job';
        ws.disconnect();
        assert.equal(ws.jobId, null);
        assert.equal(ws.isIntentionalClose, true);
    });

    describe('handleMessage', () => {
        it('dispatches status_update to onStatus callback', () => {
            const ws = new WebSocketManager();
            let received = null;
            ws.callbacks.onStatus = (data) => { received = data; };

            ws.handleMessage({ type: 'status_update', progress: 0.5 });
            assert.deepEqual(received, { type: 'status_update', progress: 0.5 });
        });

        it('dispatches metrics to onMetrics callback', () => {
            const ws = new WebSocketManager();
            let received = null;
            ws.callbacks.onMetrics = (data) => { received = data; };

            ws.handleMessage({ type: 'metrics', loss: 0.5 });
            assert.deepEqual(received, { type: 'metrics', loss: 0.5 });
        });

        it('dispatches completed and sets intentional close', () => {
            const ws = new WebSocketManager();
            let received = null;
            ws.callbacks.onCompleted = (data) => { received = data; };

            ws.handleMessage({ type: 'completed', model_path: '/models/test' });
            assert.ok(received);
            assert.equal(ws.isIntentionalClose, true);
        });

        it('dispatches error to onError callback', () => {
            const ws = new WebSocketManager();
            let received = null;
            ws.callbacks.onError = (msg) => { received = msg; };

            ws.handleMessage({ type: 'error', message: 'OOM' });
            assert.equal(received, 'OOM');
        });

        it('handles ping type without error', () => {
            const ws = new WebSocketManager();
            // Should not throw
            ws.handleMessage({ type: 'ping' });
        });

        it('handles unknown type without error', () => {
            const ws = new WebSocketManager();
            // Should not throw
            ws.handleMessage({ type: 'some_future_type', data: {} });
        });
    });
});
