// dreamkit-page.js — shared helpers for plugin pages (PLUGIN_STANDARD.md §7a), served at /theme/dreamkit-page.js.
// A page calls dk.page(plugin, title, build): it loads the run the panel passed (?run_id=&attempt=), its run.json and
// events.jsonl, and renders what build({run, episodes, events, mine}) returns. Colours come only from the theme's
// classes (.dk-chart .ok/.bad/…); text from data goes in through textContent, never innerHTML.
;(() => {
  const q = new URLSearchParams(location.search)
  const RUN = q.get('run_id') || ''
  const ATTEMPT = q.get('attempt') == null ? null : Number(q.get('attempt'))
  const NS = (() => { const x = document.createElement('div'); x.innerHTML = '<svg></svg>'; return x.firstChild.namespaceURI })()

  function h(tag, attrs = {}, ...kids) {
    const el = attrs.svg ? document.createElementNS(NS, tag) : document.createElement(tag)
    for (const [k, v] of Object.entries(attrs)) if (k !== 'svg' && v != null && v !== false) el.setAttribute(k, v === true ? '' : v)
    for (const k of kids.flat(Infinity)) if (k != null && k !== false) el.append(k instanceof Node ? k : String(k))
    return el
  }
  const s = (tag, attrs = {}, ...kids) => h(tag, { ...attrs, svg: 1 }, ...kids)
  const media = (path) => `/media/${encodeURIComponent(RUN)}/${path}`

  async function getJson(path) {
    const r = await fetch(media(path), { cache: 'no-store' })
    if (!r.ok) throw new Error(`${path}: ${r.status}`)
    return r.json()
  }
  async function getEvents() {
    try {
      const r = await fetch(media('events.jsonl'), { cache: 'no-store' })
      if (!r.ok) return []
      return (await r.text()).split('\n').filter(Boolean).map((l) => { try { return JSON.parse(l) } catch { return null } })
        .filter((e) => e && e.type === 'event')
    } catch { return [] }
  }

  const fmt = (v, d = 1) => (v == null || Number.isNaN(Number(v)) ? '–' : Number(v).toFixed(d))
  const pct = (k, n) => (n ? `${Math.round((100 * k) / n)}%` : '–')

  /** Big "impact" numbers: [{label, value, sub, tone: 'ok'|'bad'|'warn'}] */
  function stats(items) {
    return h('div', { class: 'dk-stats' }, items.map((it) => h('div', { class: 'dk-card dk-stat' },
      h('small', {}, it.label), h('div', { class: `dk-big${it.tone ? ' dk-' + it.tone : ''}` }, it.value), it.sub ? h('small', {}, it.sub) : null)))
  }
  const card = (title, ...body) => h('section', { class: 'dk-card' }, h('h2', { class: 'dk-title' }, title), h('div', { class: 'dk-body' }, body))
  const empty = (text) => h('p', { class: 'dk-empty' }, text)
  const row = (...kids) => h('div', { class: 'dk-row' }, kids)

  /** Vertical bars: [{label, value, tone, title}] */
  function bars(data, { max, unit = '', height = 120 } = {}) {
    if (!data.length) return empty('Nothing to chart yet.')
    const W = 560, H = height, top = Math.max(max ?? 0, ...data.map((d) => d.value), 1e-9), gap = 6
    const bw = Math.max(4, Math.min(48, (W - 36) / data.length - gap)), every = Math.ceil(data.length / 16)
    return s('svg', { class: 'dk-chart', viewBox: `0 0 ${W} ${H + 20}`, role: 'img' },
      s('line', { class: 'axis', x1: 30, x2: W, y1: H, y2: H }),
      s('text', { x: 0, y: 10 }, `${fmt(top, top < 10 ? 1 : 0)}${unit}`), s('text', { x: 18, y: H }, '0'),
      data.map((d, i) => {
        const x = 34 + i * (bw + gap), bh = d.value > 0 ? Math.max(2, (d.value / top) * (H - 8)) : 0
        return [s('rect', { class: d.tone || 'info', x, y: H - bh, width: bw, height: bh, rx: 2 }, s('title', {}, d.title || `${d.label}: ${d.value}${unit}`)),
          i % every === 0 ? s('text', { x: x + bw / 2, y: H + 14, 'text-anchor': 'middle' }, d.label) : null]
      }))
  }

  /** Horizontal bars with counts: [{label, value, tone}] */
  function hbars(data) {
    if (!data.length) return empty('Nothing to chart yet.')
    const W = 560, row = 24, top = Math.max(...data.map((d) => d.value), 1)
    return s('svg', { class: 'dk-chart', viewBox: `0 0 ${W} ${data.length * row}`, role: 'img' },
      data.map((d, i) => {
        const y = i * row, w = Math.max(3, ((W - 210) * d.value) / top)
        return [s('text', { x: 0, y: y + 15 }, d.label.length > 28 ? d.label.slice(0, 27) + '…' : d.label),
          s('rect', { class: d.tone || 'cat', x: 180, y: y + 4, width: w, height: 14, rx: 3 }),
          s('text', { x: 186 + w, y: y + 15 }, String(d.value))]
      }))
  }

  /** One row per attempt on a shared time axis: [{label, duration, tone, marks: [{t, tone, title}]}] */
  function timeline(rows) {
    if (!rows.length) return empty('No attempts yet.')
    const W = 560, row = 26, left = 70, span = Math.max(...rows.map((r) => r.duration || 0), 1)
    const x = (t) => left + ((W - left - 8) * Math.min(t, span)) / span
    return s('svg', { class: 'dk-chart', viewBox: `0 0 ${W} ${rows.length * row + 18}`, role: 'img' },
      rows.map((r, i) => {
        const y = i * row + 12
        return [s('text', { x: 0, y: y + 4 }, r.label),
          s('line', { class: 'track', x1: left, x2: x(r.duration || span), y1: y, y2: y }),
          s('circle', { class: r.tone || 'muted', cx: x(r.duration || span), cy: y, r: 4 }, s('title', {}, `${r.label} ended at ${fmt(r.duration)} s`)),
          (r.marks || []).map((m) => s('circle', { class: m.tone || 'warn', cx: x(m.t), cy: y, r: 5 }, s('title', {}, `${fmt(m.t)} s: ${m.title}`)))]
      }),
      s('text', { x: left, y: rows.length * row + 16 }, '0 s'), s('text', { x: W, y: rows.length * row + 16, 'text-anchor': 'end' }, `${fmt(span, 0)} s`))
  }

  /** A table: columns [{key, label, fmt?}], rows of objects (a row with _here is the attempt the panel opened). */
  function table(columns, rows) {
    return h('table', { class: 'dk-table' }, h('thead', {}, h('tr', {}, columns.map((c) => h('th', {}, c.label)))),
      h('tbody', {}, rows.map((r) => h('tr', { class: r._here ? 'dk-here' : null }, columns.map((c) => h('td', {}, c.fmt ? c.fmt(r[c.key], r) : (r[c.key] ?? '–')))))))
  }

  /** "Download JSON": a blob made inside the sandbox (a /media link would just navigate). */
  function downloadButton(name, data) {
    const b = h('button', { class: 'dk-btn' }, 'Download JSON')
    b.onclick = () => {
      const url = URL.createObjectURL(new Blob([JSON.stringify(data(), null, 1)], { type: 'application/json' }))
      const a = h('a', { href: url, download: `${name}_${RUN}.json` })
      document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000)
    }
    return b
  }

  const NO_RUN = 'No run yet. Start one, e.g. python examples/sim_loop.py --mock --episodes 2, then open this page again.'

  /** The page: a header, then whatever build() returns; reloads every 4 s while the run is live. */
  async function page(plugin, title, build) {
    const root = h('main', { class: 'dk-page' })
    document.body.append(root)
    let last = '', scrolled = false
    async function render() {
      if (!RUN) { root.replaceChildren(h('h1', { class: 'dk-h1' }, title), empty(NO_RUN)); return false }
      let run
      try { run = await getJson('run.json') } catch { root.replaceChildren(h('h1', { class: 'dk-h1' }, title), empty(NO_RUN)); return false }
      const key = `${run.updated}|${(run.episodes || []).length}|${run.status}`
      if (key === last) return run.status === 'running'
      last = key
      const events = await getEvents()
      const episodes = (run.episodes || []).map((e) => ({ ...e, _here: e.episode === ATTEMPT }))
      const on = (run.plugins || []).some((p) => (typeof p === 'string' ? p : p.name) === plugin)
      const head = h('div', { class: 'dk-headrow' }, h('h1', { class: 'dk-h1' }, title, ' ', h('span', { class: 'dk-faint' }, run.id || RUN)),
        run.mock ? h('span', { class: 'dk-pill warn' }, 'mock: scripted failures') : null)
      let body
      try {
        body = !on ? [empty(`${plugin} was off in this run. Switch it on on the Plugins page and run again.`)]
          : !episodes.length ? [empty('This run has no finished attempts yet.')]
          : build({ run, episodes, events, mine: events.filter((e) => e.plugin === plugin) })
      } catch (err) { body = [empty(`Could not draw this page: ${err.message}`)] }
      root.replaceChildren(head, ...[body].flat())
      const here = document.querySelector('.dk-here')
      if (here && !scrolled) { here.scrollIntoView({ block: 'center' }); scrolled = true }
      return run.status === 'running'
    }
    const tick = async () => { if (await render()) setTimeout(tick, 4000) }
    tick()
  }

  window.dk = { h, s, media, fmt, pct, stats, card, empty, row, bars, hbars, timeline, table, downloadButton, page, RUN, ATTEMPT }
})()
