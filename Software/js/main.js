/* ============================================
   T.A.L.O.N — INTERACTIONS (MINIMAL)
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initNav();
    initBurger();
    initReveal();
    initTyping();
    initCounters();
    initCursor();
});

/* ---- Custom cursor with smoky trail ---- */
function initCursor() {
    // Don't run on touch-only devices
    if ('ontouchstart' in window && !window.matchMedia('(pointer: fine)').matches) return;

    // Create cursor element
    const cursor = document.createElement('div');
    cursor.className = 'custom-cursor';
    const img = document.createElement('img');
    img.src = 'car.png';
    img.alt = '';
    img.draggable = false;
    cursor.appendChild(img);
    document.body.appendChild(cursor);

    let cx = -100, cy = -100;
    let lastSmoke = 0;
    const smokeInterval = 35; // ms between smoke spawns

    document.addEventListener('mousemove', e => {
        cx = e.clientX;
        cy = e.clientY;
        cursor.style.left = cx + 'px';
        cursor.style.top = cy + 'px';

        // Spawn smoke particles
        const now = performance.now();
        if (now - lastSmoke > smokeInterval) {
            lastSmoke = now;
            spawnSmoke(cx, cy);
        }
    });

    // Hide cursor when mouse leaves the window
    document.addEventListener('mouseleave', () => {
        cursor.style.left = '-100px';
        cursor.style.top = '-100px';
    });

    function spawnSmoke(x, y) {
        const p = document.createElement('div');
        p.className = 'smoke-particle';

        // Randomize drift direction
        const sx = (Math.random() - 0.5) * 24;
        const sy = -8 - Math.random() * 18;
        p.style.setProperty('--sx', sx + 'px');
        p.style.setProperty('--sy', sy + 'px');

        // Slightly randomize size
        const size = 6 + Math.random() * 6;
        p.style.width = size + 'px';
        p.style.height = size + 'px';

        // Position behind the car (offset down from cursor center)
        p.style.left = (x + (Math.random() - 0.5) * 10) + 'px';
        p.style.top = (y + 14 + Math.random() * 6) + 'px';

        document.body.appendChild(p);

        // Remove after animation
        p.addEventListener('animationend', () => p.remove());
        // Fallback cleanup
        setTimeout(() => { if (p.parentNode) p.remove(); }, 1000);
    }
}

/* ---- Theme toggle ---- */
function initTheme() {
    const saved = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', saved);

    document.querySelectorAll('.theme-switch').forEach(btn => {
        btn.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('theme', next);
        });
    });
}

/* ---- Navbar pin on scroll ---- */
function initNav() {
    const nav = document.querySelector('.nav');
    if (!nav) return;
    const check = () => nav.classList.toggle('pinned', window.scrollY > 30);
    window.addEventListener('scroll', check, { passive: true });
    check();
}

/* ---- Mobile burger ---- */
function initBurger() {
    const btn = document.querySelector('.burger');
    const menu = document.querySelector('.nav-menu');
    if (!btn || !menu) return;
    btn.addEventListener('click', () => {
        btn.classList.toggle('x');
        menu.classList.toggle('open');
    });
    menu.querySelectorAll('a').forEach(a =>
        a.addEventListener('click', () => {
            btn.classList.remove('x');
            menu.classList.remove('open');
        })
    );
}

/* ---- Scroll reveal ---- */
function initReveal() {
    const els = document.querySelectorAll('.rv');
    if (!els.length) return;
    const obs = new IntersectionObserver(entries => {
        entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('on'); });
    }, { threshold: 0.12, rootMargin: '0px 0px -30px 0px' });
    els.forEach(el => obs.observe(el));
}

/* ---- Typing effect ---- */
function initTyping() {
    const wrap = document.querySelector('.hero-typed-wrap');
    if (!wrap) return;
    const span = wrap.querySelector('.typed-txt');
    if (!span) return;

    const lines = [
        'Detect wildlife threats in real time.',
        'Monitor agricultural fields 24/7.',
        'Edge AI — YOLO & TFLite on-device.',
        'Stream live video over Wi-Fi.',
        'Autonomous patrol & alert system.'
    ];

    let li = 0, ci = 0, del = false;
    const speed = { t: 50, d: 28, wait: 2400 };

    function tick() {
        const cur = lines[li];
        if (!del) {
            span.textContent = cur.slice(0, ++ci);
            if (ci === cur.length) { del = true; return setTimeout(tick, speed.wait); }
        } else {
            span.textContent = cur.slice(0, --ci);
            if (ci === 0) { del = false; li = (li + 1) % lines.length; return setTimeout(tick, 350); }
        }
        setTimeout(tick, del ? speed.d : speed.t);
    }
    setTimeout(tick, 1000);
}

/* ---- Counter animation ---- */
function initCounters() {
    const els = document.querySelectorAll('[data-count]');
    if (!els.length) return;
    const obs = new IntersectionObserver(entries => {
        entries.forEach(e => {
            if (e.isIntersecting && !e.target.dataset.done) {
                e.target.dataset.done = '1';
                runCount(e.target);
            }
        });
    }, { threshold: 0.3 });
    els.forEach(el => obs.observe(el));
}

function runCount(el) {
    const end = +el.dataset.count;
    const pre = el.dataset.prefix || '';
    const suf = el.dataset.suffix || '';
    const dur = 1800;
    const t0 = performance.now();
    (function step(now) {
        const p = Math.min((now - t0) / dur, 1);
        const ease = 1 - Math.pow(1 - p, 3);
        el.textContent = pre + Math.round(ease * end).toLocaleString() + suf;
        if (p < 1) requestAnimationFrame(step);
    })(t0);
}
