/**
 * Merlina's Secret Miracle Module
 *
 * A hidden easter egg that brings extra magic to the training experience.
 * Activated by clicking the wizard hat logo 7 times (a magical number!)
 */

// Track click count on the logo
let miracleClickCount = 0;
let miracleClickTimer = null;
const MIRACLE_THRESHOLD = 7;

// Magical symbols for the effect
const MIRACLE_SYMBOLS = ['✨', '⭐', '🌟', '💫', '🔮', '🪄', '🌈', '💜', '💖', '🦄'];
const MIRACLE_MESSAGES = [
    'A miracle has occurred!',
    'Magic is real!',
    'You found the secret!',
    'Merlina blesses your training!',
    'May your gradients converge!',
    'Loss approaching zero...',
    'The spell is complete!'
];

/**
 * Create a floating star element
 */
function createMiracleStar(x, y) {
    const star = document.createElement('div');
    star.className = 'miracle-star';
    star.textContent = MIRACLE_SYMBOLS[Math.floor(Math.random() * MIRACLE_SYMBOLS.length)];
    star.style.left = `${x}px`;
    star.style.top = `${y}px`;

    // Random direction
    const angle = Math.random() * Math.PI * 2;
    const distance = 100 + Math.random() * 200;
    star.style.setProperty('--tx', `${Math.cos(angle) * distance}px`);
    star.style.setProperty('--ty', `${Math.sin(angle) * distance}px`);

    document.body.appendChild(star);

    setTimeout(() => star.remove(), 2000);
}

/**
 * Create rising particles from the bottom
 */
function createMiracleParticles() {
    for (let i = 0; i < 30; i++) {
        setTimeout(() => {
            const particle = document.createElement('div');
            particle.className = 'miracle-particle';
            particle.textContent = MIRACLE_SYMBOLS[Math.floor(Math.random() * MIRACLE_SYMBOLS.length)];
            particle.style.left = `${Math.random() * 100}vw`;
            particle.style.fontSize = `${20 + Math.random() * 30}px`;
            particle.style.animationDuration = `${3 + Math.random() * 2}s`;
            particle.style.animationDelay = `${Math.random() * 0.5}s`;

            document.body.appendChild(particle);

            setTimeout(() => particle.remove(), 5000);
        }, i * 100);
    }
}

/**
 * Create sparkle effects around the screen
 */
function createMiracleSparkles() {
    for (let i = 0; i < 20; i++) {
        setTimeout(() => {
            const sparkle = document.createElement('div');
            sparkle.className = 'miracle-sparkle';
            sparkle.style.left = `${Math.random() * 100}vw`;
            sparkle.style.top = `${Math.random() * 100}vh`;
            sparkle.style.animationDelay = `${Math.random() * 0.5}s`;

            document.body.appendChild(sparkle);

            setTimeout(() => sparkle.remove(), 1500);
        }, i * 50);
    }
}

/**
 * Show the miracle message
 */
function showMiracleMessage() {
    const message = MIRACLE_MESSAGES[Math.floor(Math.random() * MIRACLE_MESSAGES.length)];

    const text = document.createElement('div');
    text.className = 'miracle-text';
    text.textContent = message;
    document.body.appendChild(text);

    // Trigger animation
    requestAnimationFrame(() => {
        text.classList.add('active');
    });

    setTimeout(() => {
        text.classList.remove('active');
        setTimeout(() => text.remove(), 300);
    }, 2500);
}

/**
 * Create the rainbow overlay
 */
function createMiracleOverlay() {
    let overlay = document.querySelector('.miracle-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'miracle-overlay';
        document.body.appendChild(overlay);
    }

    requestAnimationFrame(() => {
        overlay.classList.add('active');
    });

    setTimeout(() => {
        overlay.classList.remove('active');
    }, 3000);
}

/**
 * The main miracle effect!
 */
function performMiracle() {
    // Get the wizard hat position for the effect origin
    const wizardHat = document.querySelector('.wizard-hat');
    const rect = wizardHat ? wizardHat.getBoundingClientRect() : { left: window.innerWidth / 2, top: 100 };
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    // Add glow effect to wizard hat
    if (wizardHat) {
        wizardHat.classList.add('miracle-active');
        setTimeout(() => wizardHat.classList.remove('miracle-active'), 3000);
    }

    // Create the magical effects
    createMiracleOverlay();

    // Burst of stars from the wizard hat
    for (let i = 0; i < 20; i++) {
        setTimeout(() => {
            createMiracleStar(centerX + (Math.random() - 0.5) * 50, centerY + (Math.random() - 0.5) * 50);
        }, i * 50);
    }

    // Rising particles from the bottom
    setTimeout(createMiracleParticles, 300);

    // Sparkles around the screen
    setTimeout(createMiracleSparkles, 500);

    // Show the message
    setTimeout(showMiracleMessage, 800);

    // Log to console for extra magic
    console.log('%c✨ A MIRACLE HAS OCCURRED! ✨',
        'color: #c042ff; font-size: 24px; font-weight: bold; text-shadow: 2px 2px 4px #ff6bb3;');
    console.log('%cMerlina smiles upon your training endeavors...',
        'color: #ff6bb3; font-size: 14px; font-style: italic;');
}

/**
 * Handle logo clicks for the easter egg
 */
function handleMiracleClick() {
    miracleClickCount++;

    // Reset timer
    if (miracleClickTimer) {
        clearTimeout(miracleClickTimer);
    }

    // Reset after 2 seconds of no clicks
    miracleClickTimer = setTimeout(() => {
        miracleClickCount = 0;
    }, 2000);

    // Small sparkle feedback on each click
    const wizardHat = document.querySelector('.wizard-hat');
    if (wizardHat) {
        const rect = wizardHat.getBoundingClientRect();
        createMiracleStar(
            rect.left + rect.width / 2 + (Math.random() - 0.5) * 30,
            rect.top + rect.height / 2 + (Math.random() - 0.5) * 30
        );
    }

    // Check for miracle threshold
    if (miracleClickCount >= MIRACLE_THRESHOLD) {
        miracleClickCount = 0;
        performMiracle();
    }
}

/**
 * Initialize the miracle easter egg
 */
export function initMiracle() {
    // Attach click handler to wizard hat logo
    const wizardHat = document.querySelector('.wizard-hat');
    if (wizardHat) {
        wizardHat.style.cursor = 'pointer';
        wizardHat.addEventListener('click', handleMiracleClick);
    }

    // Also attach to the logo text for easier triggering
    const logoH1 = document.querySelector('.logo h1');
    if (logoH1) {
        logoH1.style.cursor = 'pointer';
        logoH1.addEventListener('click', handleMiracleClick);
    }

    // Secret keyboard shortcut: typing "miracle" triggers it
    let miracleSequence = '';
    const MIRACLE_WORD = 'miracle';

    document.addEventListener('keypress', (e) => {
        miracleSequence += e.key.toLowerCase();

        // Keep only the last N characters
        if (miracleSequence.length > MIRACLE_WORD.length) {
            miracleSequence = miracleSequence.slice(-MIRACLE_WORD.length);
        }

        if (miracleSequence === MIRACLE_WORD) {
            miracleSequence = '';
            performMiracle();
        }
    });
}

// Export for manual triggering (e.g., from console)
window.performMiracle = performMiracle;
