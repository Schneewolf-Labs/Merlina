/**
 * HuggingFace namespace picker.
 *
 * A bare model name uploads to the token owner's personal account, which is
 * why org uploads used to 404. These helpers fill a <select> with the
 * namespaces a token can actually publish to (the account itself plus every
 * organization it belongs to) so the user picks instead of guessing.
 */

import { MerlinaAPI } from './api.js';

const DEFAULT_HINT = 'Click "Check Orgs" to list the organizations your token can publish to.';

/**
 * Ensure a namespace value is selectable, adding an option for values the
 * dropdown hasn't learned about yet (e.g. restored from a saved config).
 */
export function setNamespaceValue(selectId, value) {
    const select = document.getElementById(selectId);
    if (!select) return;

    const wanted = (value || '').trim();
    if (!wanted) {
        select.value = '';
        return;
    }

    const exists = Array.from(select.options).some(opt => opt.value === wanted);
    if (!exists) {
        const option = document.createElement('option');
        option.value = wanted;
        option.textContent = wanted;
        select.appendChild(option);
    }
    select.value = wanted;
}

/** Read the selected namespace, or null for the personal-account default. */
export function getNamespaceValue(selectId) {
    const select = document.getElementById(selectId);
    return select?.value?.trim() || null;
}

/**
 * Fetch the token's namespaces and populate a <select>.
 *
 * @param {Object} opts
 * @param {string} opts.selectId  ID of the <select> to fill
 * @param {string} [opts.tokenId] ID of the token input (blank uses server .env)
 * @param {string} [opts.hintId]  ID of the <small> element for status text
 * @returns {Promise<boolean>} true when the list was loaded
 */
export async function loadNamespaces({ selectId, tokenId = null, hintId = null } = {}) {
    const select = document.getElementById(selectId);
    if (!select) return false;

    const hint = hintId ? document.getElementById(hintId) : null;
    const token = tokenId ? (document.getElementById(tokenId)?.value?.trim() || null) : null;
    const previous = select.value;

    if (hint) {
        hint.textContent = 'Checking your HuggingFace namespaces…';
        hint.style.color = '#888';
    }

    try {
        const data = await MerlinaAPI.getHfNamespaces(token);
        const namespaces = Array.isArray(data?.namespaces) ? data.namespaces : [];

        select.innerHTML = '';
        const personal = document.createElement('option');
        personal.value = '';
        personal.textContent = data?.user
            ? `${data.user} (personal account)`
            : 'Personal account (default)';
        select.appendChild(personal);

        let orgCount = 0;
        namespaces.forEach(ns => {
            // The personal account is already the default option.
            if (ns.type === 'user' && ns.name === data?.user) return;
            const option = document.createElement('option');
            option.value = ns.name;
            const suffix = ns.can_write === false ? ' — read-only' : '';
            option.textContent = ns.full_name && ns.full_name !== ns.name
                ? `${ns.name} (${ns.full_name})${suffix}`
                : `${ns.name}${suffix}`;
            if (ns.can_write === false) option.dataset.readOnly = 'true';
            select.appendChild(option);
            orgCount += 1;
        });

        setNamespaceValue(selectId, previous);

        if (hint) {
            hint.style.color = orgCount ? '#2e7d32' : '#888';
            hint.textContent = orgCount
                ? `Found ${orgCount} organization${orgCount === 1 ? '' : 's'} for ${data.user}.`
                : `No organizations found for ${data.user} — uploading to your personal account.`;
        }
        return true;
    } catch (error) {
        console.warn('Failed to load HuggingFace namespaces:', error);
        if (hint) {
            hint.style.color = '#c62828';
            hint.textContent = `Could not list namespaces: ${error.message}`;
        }
        return false;
    }
}

/**
 * Wire a "Check Orgs" button to its namespace <select>.
 */
export function setupNamespacePicker({ buttonId, selectId, tokenId = null, hintId = null } = {}) {
    const button = document.getElementById(buttonId);
    if (!button || button.dataset.nsWired === 'true') return;
    button.dataset.nsWired = 'true';

    button.addEventListener('click', async () => {
        button.disabled = true;
        try {
            await loadNamespaces({ selectId, tokenId, hintId });
        } finally {
            button.disabled = false;
        }
    });
}

/** Reset a picker's hint back to its idle prompt. */
export function resetNamespaceHint(hintId) {
    const hint = document.getElementById(hintId);
    if (!hint) return;
    hint.textContent = DEFAULT_HINT;
    hint.style.color = '#888';
}
