/* MediMatch frontend logic */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const form = $('#search-form');
  const input = $('#search-input');
  const sugBox = $('#suggestions');
  const results = $('#results');
  const tabs = $('.tabs');
  const quick = $('#quick-chips');
  const modal = $('#modal');

  let mode = 'substitute';
  let sugItems = [];
  let sugIndex = -1;
  let sugTimer = null;
  let sugController = null;
  const cache = new Map(); // id -> medicine (for modal)

  const QUICK = {
    substitute: ['Panadol', 'Brufen', 'Augmentin 625 Duo Tablet', 'Risek', 'Paracetamol', 'panadoll'],
    symptom: ['headache and fever', 'acidity and heartburn', 'runny nose sneezing', 'loose motions', 'dry cough', 'high blood pressure'],
  };
  const PLACEHOLDER = {
    substitute: 'e.g. Panadol, Brufen, Augmentin, paracetamol…',
    symptom: 'Describe symptoms, e.g. headache and fever…',
  };

  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const title = (s) => s.replace(/\b\w/g, (c) => c.toUpperCase());
  const PILL_SVG = '<svg viewBox="0 0 24 24" fill="none"><rect x="3" y="8" width="18" height="8" rx="4" stroke="currentColor" stroke-width="1.8" transform="rotate(-35 12 12)"/><path d="M12 8v8" stroke="currentColor" stroke-width="1.8" transform="rotate(-35 12 12)"/></svg>';
  const INFO_SVG = '<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2"/><path d="M12 8h.01M11 12h1v4h1" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';

  /* ---------------- tabs ---------------- */
  function setMode(m) {
    mode = m;
    tabs.dataset.mode = m;
    tabs.querySelectorAll('.tab').forEach((t) => {
      const on = t.dataset.mode === m;
      t.classList.toggle('active', on);
      t.setAttribute('aria-selected', on);
    });
    input.placeholder = PLACEHOLDER[m];
    closeSug();
    renderQuick();
    input.focus();
  }
  tabs.addEventListener('click', (e) => {
    const t = e.target.closest('.tab');
    if (t) setMode(t.dataset.mode);
  });

  function renderQuick() {
    quick.innerHTML = '<span class="lbl">Try:</span>' +
      QUICK[mode].map((q, i) => `<button type="button" class="chip-btn" id="quick-${mode}-${i}" data-q="${esc(q)}">${esc(q)}</button>`).join('');
  }
  quick.addEventListener('click', (e) => {
    const b = e.target.closest('.chip-btn');
    if (!b) return;
    input.value = b.dataset.q;
    run();
  });

  /* ---------------- autocomplete ---------------- */
  input.addEventListener('input', () => {
    if (mode !== 'substitute') return;
    clearTimeout(sugTimer);
    const q = input.value.trim();
    if (q.length < 2) return closeSug();
    sugTimer = setTimeout(() => fetchSug(q), 140);
  });

  async function fetchSug(q) {
    sugController?.abort();
    sugController = new AbortController();
    try {
      const r = await fetch(`/api/suggest?q=${encodeURIComponent(q)}`, { signal: sugController.signal });
      sugItems = await r.json();
      sugIndex = -1;
      renderSug(q);
    } catch (_) { /* aborted */ }
  }

  function highlight(label, q) {
    const i = label.toLowerCase().indexOf(q.toLowerCase());
    if (i < 0) return esc(label);
    return esc(label.slice(0, i)) + '<mark>' + esc(label.slice(i, i + q.length)) + '</mark>' + esc(label.slice(i + q.length));
  }

  function renderSug(q) {
    if (!sugItems.length) return closeSug();
    sugBox.innerHTML = sugItems.map((s, i) => `
      <li role="option" data-i="${i}" id="sug-${i}" class="${i === sugIndex ? 'active' : ''}">
        <span class="s-type ${s.type}">${s.type === 'salt' ? 'Ingredient' : 'Brand'}</span>
        <span class="s-label">${highlight(s.label, q)}</span>
        <span class="s-hint">${esc(s.hint)}</span>
      </li>`).join('');
    sugBox.classList.add('open');
  }
  function closeSug() { sugBox.classList.remove('open'); sugItems = []; sugIndex = -1; }

  sugBox.addEventListener('mousedown', (e) => {
    const li = e.target.closest('li');
    if (!li) return;
    e.preventDefault();
    input.value = sugItems[+li.dataset.i].label;
    closeSug();
    run();
  });
  input.addEventListener('keydown', (e) => {
    if (!sugBox.classList.contains('open')) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      sugIndex = (sugIndex + (e.key === 'ArrowDown' ? 1 : -1) + sugItems.length) % sugItems.length;
      renderSug(input.value.trim());
    } else if (e.key === 'Enter' && sugIndex >= 0) {
      e.preventDefault();
      input.value = sugItems[sugIndex].label;
      closeSug();
      run();
    } else if (e.key === 'Escape') closeSug();
  });
  document.addEventListener('click', (e) => { if (!e.target.closest('.input-wrap')) closeSug(); });

  /* ---------------- search ---------------- */
  form.addEventListener('submit', (e) => { e.preventDefault(); closeSug(); run(); });

  async function run() {
    const q = input.value.trim();
    if (!q) return input.focus();
    results.innerHTML = $('#tpl-loading').innerHTML;
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    const url = mode === 'substitute' ? `/api/substitutes?q=${encodeURIComponent(q)}` : `/api/symptoms?q=${encodeURIComponent(q)}`;
    try {
      const r = await fetch(url);
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Request failed');
      mode === 'substitute' ? renderSubstitutes(data) : renderSymptoms(data);
    } catch (err) {
      results.innerHTML = `<div class="empty"><div class="big">😵</div><p>${esc(err.message)}</p></div>`;
    }
  }

  /* ---------------- render helpers ---------------- */
  function thumb(m) {
    const img = m.image
      ? `<img src="${esc(m.image)}" alt="${esc(m.name)}" loading="lazy" referrerpolicy="no-referrer" onload="this.naturalWidth>1&&this.classList.add('ok')" onerror="this.remove()">`
      : '';
    return `<div class="thumb">${PILL_SVG}${img}</div>`;
  }
  function saltChips(salts, targetSalts) {
    return `<div class="salts">${salts.map((s) => `<span class="salt ${targetSalts && !targetSalts.includes(s) ? 'other' : ''}">${esc(title(s))}</span>`).join('')}</div>`;
  }
  function reviewMeter(r) {
    const total = r.excellent + r.average + r.poor;
    if (!total) return '<div class="meter"><span>No user reviews</span></div>';
    return `<div class="meter"><span>Reviews</span><div class="bar"><i class="ex" style="width:${r.excellent}%"></i><i class="av" style="width:${r.average}%"></i><i class="po" style="width:${r.poor}%"></i></div><span>${r.excellent}% 👍</span></div>`;
  }
  function card(m, { targetSalts, kind, delay }) {
    cache.set(m.id, m);
    let badge = '';
    if (m.source && m.source.startsWith('Local')) badge = '<span class="badge local">PK brand</span>';
    else if (kind === 'exact') badge = m.same_strength === false ? '<span class="badge diff">Diff. strength</span>' : m.same_strength ? '<span class="badge">Same strength</span>' : '';
    const sim = kind !== 'exact' ? `<div class="sim"><span>Similarity</span><div class="track"><i style="width:${Math.min(100, Math.round(m.similarity * 100))}%"></i></div><span>${Math.round(m.similarity * 100)}%</span></div>` : '';
    return `
      <article class="card glass" data-id="${m.id}" id="med-${m.id}" style="animation-delay:${delay}ms" tabindex="0">
        ${badge}
        <div class="card-top">${thumb(m)}<div><h4>${esc(m.name)}</h4><div class="mfr">${esc(m.manufacturer)}</div></div></div>
        ${saltChips(m.salts, targetSalts)}
        ${m.reason ? `<div class="reason">↳ ${esc(m.reason)}</div>` : ''}
        <div class="uses">${esc(m.uses.join(' · ') || '—')}</div>
        ${sim}
        ${reviewMeter(m.reviews)}
      </article>`;
  }
  function grid(list, opts, id, pageSize = 9) {
    const first = list.slice(0, pageSize).map((m, i) => card(m, { ...opts, delay: i * 40 })).join('');
    const more = list.length > pageSize ? `<button class="more-btn" id="more-${id}" data-grid="${id}">Show ${list.length - pageSize} more</button>` : '';
    setTimeout(() => {
      const btn = document.getElementById(`more-${id}`);
      btn?.addEventListener('click', () => {
        document.getElementById(`grid-${id}`).insertAdjacentHTML('beforeend', list.slice(pageSize).map((m, i) => card(m, { ...opts, delay: i * 30 })).join(''));
        btn.remove();
      });
    });
    return `<div class="grid" id="grid-${id}">${first}</div>${more}`;
  }

  function renderSubstitutes(d) {
    if (!d.target) {
      results.innerHTML = `<div class="empty"><div class="big">🔍</div><p>No medicine matched <b>“${esc(d.query)}”</b>. Check the spelling or try the generic name (e.g. paracetamol).</p></div>`;
      return;
    }
    const t = d.target;
    let html = '';
    if (d.corrected) html += `<div class="notice">${INFO_SVG}<span>Showing results for <b>${esc(d.matched)}</b> (corrected from “${esc(d.query)}”, ${Math.round(d.match_score)}% match)</span></div>`;
    if (t.source && t.source.startsWith('Local')) html += `<div class="notice info">${INFO_SVG}<span><b>${esc(t.name)}</b> is a Pakistani brand. Substitutes come from the 1mg dataset, so check what's available at your local pharmacy.</span></div>`;

    html += `
      <div class="target glass">
        ${t.is_salt ? `<div class="thumb">${PILL_SVG}</div>` : thumb(t)}
        <div>
          <div class="eyebrow">${t.is_salt ? 'Active ingredient' : 'You searched for'}</div>
          <h2>${esc(t.name)}</h2>
          ${saltChips(t.salts)}
        </div>
        <div class="count"><b>${d.exact.length}${d.exact.length >= 24 ? '+' : ''}</b><small>direct substitutes</small></div>
      </div>`;

    html += `<div class="section-head"><h3>Same active ingredient <span class="tag exact">EXACT MATCH</span></h3><p>${t.is_salt ? `Medicines containing only ${esc(t.name)}` : 'Identical salt composition, so these can be used in place of it'}</p></div>`;
    html += d.exact.length ? grid(d.exact, { targetSalts: t.salts, kind: 'exact' }, 'exact')
      : `<div class="empty" style="padding:24px 0 40px"><p>No medicine in the dataset has exactly the same composition. See the alternatives below.</p></div>`;

    if (d.similar.length) {
      html += `<div class="section-head"><h3>Similar alternatives <span class="tag similar">TF-IDF</span></h3><p>${t.is_salt ? 'Combination medicines that include this ingredient' : 'Ranked by cosine similarity of ingredients + uses'}</p></div>`;
      html += grid(d.similar, { targetSalts: t.salts, kind: 'similar' }, 'similar', 6);
    }
    results.innerHTML = html;
  }

  function renderSymptoms(d) {
    let html = `<div class="terms"><span class="lbl">Expanded query terms:</span>${d.terms.map((t) => `<span class="term">${esc(t)}</span>`).join('')}</div>`;
    if (!d.results.length) {
      html += `<div class="empty"><div class="big">🤔</div><p>No medicines matched those symptoms. Try simpler words like “fever”, “cough” or “acidity”.</p></div>`;
    } else {
      html += `<div class="section-head"><h3>Suggested for “${esc(d.query)}” <span class="tag similar">TF-IDF</span></h3><p>One top-rated option per composition, ranked by relevance</p></div>`;
      html += grid(d.results, { kind: 'symptom' }, 'sym', 12);
    }
    results.innerHTML = html;
  }

  /* ---------------- modal ---------------- */
  results.addEventListener('click', (e) => {
    const c = e.target.closest('.card');
    if (c) openModal(cache.get(+c.dataset.id));
  });
  results.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.classList.contains('card')) openModal(cache.get(+e.target.dataset.id));
  });

  function openModal(m) {
    if (!m) return;
    const r = m.reviews;
    const row = (lbl, v, col) => `<div class="review-row"><span>${lbl}</span><div class="track"><i style="width:${v}%;background:${col}"></i></div><span>${v}%</span></div>`;
    $('#modal-body').innerHTML = `
      <div class="m-head">${thumb(m)}<div><h2 id="modal-title">${esc(m.name)}</h2><p>${esc(m.manufacturer)} · ${esc(m.source)}</p></div></div>
      <div class="m-sec"><h4>Composition</h4><p>${esc(m.composition)}</p></div>
      <div class="m-sec"><h4>Uses</h4><div class="list-chips ok">${m.uses.map((u) => `<span>${esc(u)}</span>`).join('') || '<span>Not available</span>'}</div></div>
      <div class="m-sec"><h4>Possible side effects</h4><div class="list-chips fx">${m.side_effects.map((u) => `<span>${esc(u)}</span>`).join('') || '<span>Not available</span>'}</div></div>
      ${r.excellent + r.average + r.poor ? `<div class="m-sec"><h4>User reviews</h4><div class="review-rows">${row('Excellent', r.excellent, 'var(--mint)')}${row('Average', r.average, 'var(--amber)')}${row('Poor', r.poor, 'var(--rose)')}</div></div>` : ''}
      <div class="m-actions"><button class="btn-ghost" id="modal-find-subs">Find substitutes for this →</button></div>`;
    $('#modal-find-subs').onclick = () => { closeModal(); setMode('substitute'); input.value = m.name; run(); };
    modal.classList.add('open');
    modal.setAttribute('aria-hidden', 'false');
  }
  function closeModal() { modal.classList.remove('open'); modal.setAttribute('aria-hidden', 'true'); }
  modal.addEventListener('click', (e) => { if (e.target.hasAttribute('data-close')) closeModal(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

  renderQuick();
})();
