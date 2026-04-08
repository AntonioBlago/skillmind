// SkillMind — skill-mind.com

// Copy to clipboard for code blocks
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const code = btn.closest('.hero__install, .step__code')?.textContent?.replace('$', '').replace('Copy', '').trim();
            if (code) {
                navigator.clipboard.writeText(code).then(() => {
                    const orig = btn.textContent;
                    btn.textContent = '✓';
                    setTimeout(() => btn.textContent = orig, 1500);
                });
            }
        });
    });

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(a => {
        a.addEventListener('click', e => {
            e.preventDefault();
            const target = document.querySelector(a.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // Nav background opacity on scroll
    const nav = document.querySelector('.nav');
    if (nav) {
        window.addEventListener('scroll', () => {
            nav.style.background = window.scrollY > 50
                ? 'rgba(10, 11, 15, 0.95)'
                : 'rgba(10, 11, 15, 0.85)';
        });
    }

    // Mobile menu toggle
    const toggle = document.querySelector('.nav__toggle');
    const links = document.querySelector('.nav__links');
    if (toggle && links) {
        toggle.addEventListener('click', () => {
            links.classList.toggle('nav__links--open');
        });
    }
});
