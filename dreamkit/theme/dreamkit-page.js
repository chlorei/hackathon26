// dreamkit-page.js — the shared plugin page (PLUGIN_STANDARD.md §7a), served at /theme/dreamkit-page.js.
// Every plugin page answers one question: what is this plugin, and what did IT do in this run?
//   dk.plugin({ how, numbers, noticed, context, extra, quiet, events, item, download })
// draws, in this order: header (icon, name, what it does, its status in the run) · How it works · 1–3 numbers ·
// What it noticed in this run (its events by attempt, each on the attempt's timeline) · extra · Settings.
// The server puts the plugin's own record in window.DK_PLUGIN (description, icon, params, defaults, schema).
// Colours only through the theme's classes; data goes in through textContent, never innerHTML.
;(() => {
  const q = new URLSearchParams(location.search)
  const RUN = q.get('run_id') || ''
  const ATTEMPT = q.get('attempt') == null ? null : Number(q.get('attempt'))
  const INFO = window.DK_PLUGIN || {}
  const NS = (() => { const x = document.createElement('div'); x.innerHTML = '<svg></svg>'; return x.firstChild.namespaceURI })()

  // -- small helpers ----------------------------------------------------------------------------------------------
  function h(tag, attrs = {}, ...kids) {
    const el = attrs.svg ? document.createElementNS(NS, tag) : document.createElement(tag)
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'svg' || v == null || v === false) continue
      if (k === 'on') for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn)
      else el.setAttribute(k, v === true ? '' : v)
    }
    for (const k of kids.flat(Infinity)) if (k != null && k !== false) el.append(k instanceof Node ? k : String(k))
    return el
  }
  const s = (tag, attrs = {}, ...kids) => h(tag, { ...attrs, svg: 1 }, ...kids)
  const media = (path) => `/media/${encodeURIComponent(RUN)}/${path}`
  const fmt = (v, d = 1) => (v == null || Number.isNaN(Number(v)) ? '–' : Number(v).toFixed(d))
  const pct = (k, n) => (n ? `${Math.round((100 * k) / n)}%` : '–')
  const plural = (n, one, many = one + 's') => `${n} ${n === 1 ? one : many}`
  const ep3 = (n) => String(n).padStart(3, '0')

  async function getJson(path) {
    const r = await fetch(media(path), { cache: 'no-store' })
    if (!r.ok) throw new Error(`${path}: ${r.status}`)
    return r.json()
  }
  const json = (path) => getJson(path).catch(() => null)
  async function getEvents() {
    try {
      const r = await fetch(media('events.jsonl'), { cache: 'no-store' })
      if (!r.ok) return []
      return (await r.text()).split('\n').filter(Boolean).map((l) => { try { return JSON.parse(l) } catch { return null } })
        .filter((e) => e && e.type === 'event')
    } catch { return [] }
  }

  // -- building blocks --------------------------------------------------------------------------------------------
  const card = (title, ...body) => h('section', { class: 'dk-card' }, h('h2', { class: 'dk-title' }, title), h('div', { class: 'dk-body' }, body))
  const empty = (text) => h('p', { class: 'dk-empty' }, text)
  const row = (...kids) => h('div', { class: 'dk-row' }, kids)

  /** 1–3 numbers about this plugin: [{label, value, sub, tone}] */
  const stats = (items) => h('div', { class: 'dk-stats' }, items.slice(0, 3).map((it) => h('div', { class: 'dk-card dk-stat' },
    h('small', {}, it.label), h('div', { class: `dk-big${it.tone ? ' dk-' + it.tone : ''}` }, it.value), it.sub ? h('small', {}, it.sub) : null)))

  let ICONS = null
  async function icon(name) {
    ICONS = ICONS || (await fetch('/theme/icons.json').then((r) => r.json()).catch(() => ({})))
    const node = ICONS[name] || ICONS.Plug || []
    return s('svg', { class: 'dk-icon', viewBox: '0 0 24 24', 'aria-hidden': 'true' }, node.map(([tag, a]) => s(tag, a)))
  }

  const isInt = (vals) => vals.every((v) => Number.isInteger(v))
  /** Vertical bars: [{label, value, tone, title}]; counts get whole-number axes. */
  function bars(data, { max, unit = '', height = 110 } = {}) {
    if (!data.length) return empty('Nothing to chart yet.')
    const W = 560, H = height, gap = 6, whole = isInt(data.map((d) => d.value))
    let top = Math.max(max ?? 0, ...data.map((d) => d.value), 1e-9)
    if (whole) top = Math.max(1, Math.ceil(top))
    const bw = Math.max(4, Math.min(48, (W - 36) / data.length - gap)), every = Math.ceil(data.length / 16)
    return s('svg', { class: 'dk-chart', viewBox: `0 0 ${W} ${H + 20}`, role: 'img' },
      s('line', { class: 'axis', x1: 30, x2: W, y1: H, y2: H }),
      s('text', { x: 0, y: 10 }, `${whole ? top : fmt(top, top < 1 ? 2 : top < 10 ? 1 : 0)}${unit}`), s('text', { x: 18, y: H }, '0'),
      data.map((d, i) => {
        const x = 34 + i * (bw + gap), bh = d.value > 0 ? Math.max(2, (d.value / top) * (H - 8)) : 0
        return [s('rect', { class: d.tone || 'info', x, y: H - bh, width: bw, height: bh, rx: 2 }, s('title', {}, d.title || `${d.label}: ${d.value}${unit}`)),
          i % every === 0 ? s('text', { x: x + bw / 2, y: H + 14, 'text-anchor': 'middle' }, d.label) : null]
      }))
  }
  /** Horizontal bars with counts: [{label, value, tone}] */
  function hbars(data) {
    if (!data.length) return empty('Nothing to chart yet.')
    const W = 560, rowH = 24, top = Math.max(...data.map((d) => d.value), 1)
    return s('svg', { class: 'dk-chart', viewBox: `0 0 ${W} ${data.length * rowH}`, role: 'img' },
      data.map((d, i) => {
        const y = i * rowH, w = Math.max(3, ((W - 210) * d.value) / top)
        return [s('text', { x: 0, y: y + 15 }, d.label.length > 28 ? d.label.slice(0, 27) + '…' : d.label),
          s('rect', { class: d.tone || 'cat', x: 180, y: y + 4, width: w, height: 14, rx: 3 }),
          s('text', { x: 186 + w, y: y + 15 }, String(d.value))]
      }))
  }
  /** A table: columns [{key, label, fmt?}], rows of objects. */
  const table = (columns, rows) => h('table', { class: 'dk-table' }, h('thead', {}, h('tr', {}, columns.map((c) => h('th', {}, c.label)))),
    h('tbody', {}, rows.map((r) => h('tr', {}, columns.map((c) => h('td', {}, c.fmt ? c.fmt(r[c.key], r) : (r[c.key] ?? '–')))))))

  /** "Download JSON": a blob made inside the sandbox (a /media link would just navigate). */
  function downloadButton(name, data) {
    return h('button', { class: 'dk-btn', on: { click: () => {
      const url = URL.createObjectURL(new Blob([JSON.stringify(data(), null, 1)], { type: 'application/json' }))
      const a = h('a', { href: url, download: `${name}_${RUN || 'no-run'}.json` })
      document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000)
    } } }, 'Download JSON')
  }

  const TONE = { success: 'ok', error: 'bad', warn: 'warn', info: 'info' }
  const dotClass = (level) => `dk-dot ${({ success: 'ok', error: 'bad', warn: 'warn', info: 'run' })[level] || ''}`
  /** One attempt: header, its timeline with this plugin's moments, the moments as a list (click one to find it). */
  function attemptBlock(ep, items, context) {
    const W = 560, left = 6, dur = Math.max(ep.duration_s || 0, ...items.map((i) => i.t || 0), 1)
    const x = (t) => left + ((W - 2 * left) * Math.min(Math.max(t, 0), dur)) / dur
    const dots = [], rows = []
    const select = (i) => { dots.forEach((d, j) => d && d.classList.toggle('sel', i === j)); rows.forEach((r, j) => r.classList.toggle('sel', i === j)) }
    items.forEach((it, i) => {
      if (it.t != null) dots[i] = s('circle', { class: TONE[it.level] || 'info', cx: x(it.t), cy: 12, r: 5, on: { click: () => select(i) } }, s('title', {}, `${fmt(it.t)} s · ${it.text}`))
      // the panel opens the attempt's video at this second (a sandboxed page cannot navigate it; it asks)
      const watch = it.t != null && (ep.videos || []).length
        ? h('button', { class: 'dk-watch', title: `Watch attempt ${ep.episode} from ${fmt(Math.max(0, it.t - 1))} s`,
            on: { click: (e) => { e.stopPropagation(); parent.postMessage({ type: 'dreamkit:watch', run: RUN, episode: ep.episode, t: it.t }, '*') } } }, '▶ watch')
        : h('span')
      rows[i] = h('li', { class: 'dk-ev', tabindex: 0, on: { click: () => select(i), keydown: (e) => { if (e.key === 'Enter') select(i) } } },
        h('span', { class: 'dk-ev-t' }, it.t == null ? '' : `${fmt(it.t)} s`), h('span', { class: dotClass(it.level) }),
        h('span', { class: `dk-ev-m${it.level === 'error' ? ' dk-bad' : it.level === 'warn' ? ' dk-warn' : ''}` }, it.text), watch)
    })
    const tl = s('svg', { class: 'dk-chart dk-tl', viewBox: `0 0 ${W} 32`, role: 'img', 'aria-label': `attempt ${ep.episode} timeline` },
      s('line', { class: 'track', x1: x(0), x2: x(ep.duration_s || dur), y1: 12, y2: 12 }),
      dots.filter(Boolean), s('text', { x: x(0), y: 30 }, '0 s'), s('text', { x: x(dur), y: 30, 'text-anchor': 'end' }, `${fmt(dur)} s`))
    const outcome = ep.success === true ? h('span', { class: 'dk-pill ok' }, 'success') : ep.success === false ? h('span', { class: 'dk-pill bad' }, 'failed') : h('span', { class: 'dk-pill' }, 'unknown')
    return h('div', { class: `dk-attempt${ep.episode === ATTEMPT ? ' dk-here' : ''}`, id: `attempt-${ep.episode}` },
      h('div', { class: 'dk-attempt-head' }, h('b', {}, `Attempt ${ep.episode}`), outcome, h('span', { class: 'dk-faint' }, `${fmt(ep.duration_s)} s`)),
      items.length ? [tl, h('ul', { class: 'dk-evs' }, rows)] : null, context || null)
  }

  function settings(name) {
    const props = (INFO.schema && INFO.schema.properties) || {}
    const params = INFO.params || {}, defaults = INFO.defaults || {}
    const show = (v) => (v == null || v === '' || (Array.isArray(v) && !v.length) ? '—' : Array.isArray(v) ? v.join(', ') : typeof v === 'object' ? Object.entries(v).map(([k, x]) => `${k}: ${x}`).join(', ') : String(v))
    const open = h('button', { class: 'dk-btn', on: { click: () => parent.postMessage({ type: 'dreamkit:open-settings', plugin: name }, '*') } }, 'Change settings')
    const keys = Object.keys(props)
    return card('Settings', keys.length ? h('dl', { class: 'dk-settings' }, keys.map((k) => {
      const v = k in params ? params[k] : defaults[k], unit = props[k].unit ? ` ${props[k].unit}` : ''
      const changed = k in params && JSON.stringify(params[k]) !== JSON.stringify(defaults[k])
      return [h('dt', {}, k), h('dd', {}, h('b', {}, show(v) + unit), changed ? h('span', { class: 'dk-faint' }, ` (default ${show(defaults[k])})`) : null,
        h('div', { class: 'dk-faint' }, props[k].description || ''))]
    })) : empty('This plugin has no settings.'), row(open))
  }

  const NO_RUN = () => h('p', { class: 'dk-empty' }, 'Run sim_loop to see what this plugin does: ', h('code', {}, 'python examples/sim_loop.py --mock --episodes 3'))

  /** The page. spec: { how(P, info) → text|nodes, numbers(c) → [..], noticed(c) → nodes, context(c, ep) → node,
   *  extra(c) → nodes, quiet(c) → text, events(e) → bool, item(e) → {t, level, text}, download(c) → data } */
  function plugin(spec = {}) {
    const name = spec.name || INFO.name || ''
    const root = h('main', { class: 'dk-page' })
    document.body.append(root)
    const P = { ...(INFO.defaults || {}), ...(INFO.params || {}) }
    let last = '', scrolled = false

    async function header(run) {
      const st = run && run.plugin_stats ? run.plugin_stats[name] : null
      const on = run ? (run.plugins || []).some((p) => (typeof p === 'string' ? p : p.name) === name) : null
      const chips = [
        on == null ? h('span', { class: 'dk-pill' }, 'no run yet') : on ? h('span', { class: 'dk-pill ok' }, 'on in this run') : h('span', { class: 'dk-pill' }, 'off in this run'),
        st && st.ms_per_step != null ? h('span', { class: 'dk-chip' }, `${fmt(st.ms_per_step, 2)} ms/step`) : null,
        st ? h('span', { class: `dk-chip${st.errors ? ' dk-bad' : ''}` }, plural(st.errors || 0, 'error')) : null,
        st && st.disabled ? h('span', { class: 'dk-pill bad' }, `switched off: ${st.disabled}`) : null,
      ]
      return h('header', { class: 'dk-plugin-head' }, h('div', { class: 'dk-icon-box' }, await icon(INFO.icon)),
        h('div', { class: 'dk-plugin-text' }, h('h1', { class: 'dk-plugin-name' }, name), h('p', { class: 'dk-plugin-what' }, INFO.description || ''),
          h('div', { class: 'dk-row' }, chips),
          run ? h('p', { class: 'dk-run-id' }, `run ${run.id || RUN}${run.mock ? ' · mock: scripted failures' : ''}`) : null))
    }

    const how = () => card('How it works', (spec.how && spec.how(P, INFO)) || h('p', {}, INFO.description || 'No description.'))

    function noticed(c) {
      if (spec.noticed) return spec.noticed(c)
      const item = spec.item || ((e) => ({ t: e.t_s, level: e.level, text: e.message }))
      const blocks = [], quiet = []
      for (const ep of c.episodes) {
        const items = c.mine.filter((e) => e.episode === ep.episode).map(item)
        const extra = spec.context ? spec.context(c, ep) : null
        if (items.length || extra) blocks.push(attemptBlock(ep, items, extra))
        else quiet.push(ep.episode)
      }
      const runLevel = c.mine.filter((e) => !c.episodes.some((ep) => ep.episode === e.episode)).map(item)
      if (!blocks.length && !runLevel.length) {
        return card('What it noticed in this run', empty(spec.quiet ? spec.quiet(c) : `Nothing to report: all ${plural(c.episodes.length, 'attempt')} looked fine to it.`))
      }
      return card('What it noticed in this run',
        runLevel.length ? h('ul', { class: 'dk-evs' }, runLevel.map((it) => h('li', { class: 'dk-ev' }, h('span', { class: 'dk-ev-t' }, 'run'),
          h('span', { class: dotClass(it.level) }), h('span', { class: 'dk-ev-m' }, it.text)))) : null,
        blocks, quiet.length ? h('p', { class: 'dk-faint dk-quiet' }, `Nothing in ${quiet.length === 1 ? 'attempt' : 'attempts'} ${quiet.join(', ')}.`) : null,
        spec.download === false ? null : row(downloadButton(name, () => (spec.download ? spec.download(c) : c.mine))))
    }

    async function render() {
      const run = RUN ? await json('run.json') : null
      const key = run ? `${run.updated}|${(run.episodes || []).length}|${run.status}` : 'none'
      if (key === last) return Boolean(run && run.status === 'running')
      last = key
      const parts = [await header(run), how()]
      if (!run) {
        parts.push(card('What it noticed in this run', NO_RUN()))
      } else {
        const events = await getEvents()
        const episodes = run.episodes || []
        const on = (run.plugins || []).some((p) => (typeof p === 'string' ? p : p.name) === name)
        const mine = events.filter(spec.events || ((e) => e.plugin === name))
        const c = { run, episodes, events, mine, P, info: INFO, on }
        try {
          if (!on) parts.push(card('What it noticed in this run', empty(`${name} was off in this run. Switch it on (Change settings below, or the Plugins page) and run again.`)))
          else if (!episodes.length) parts.push(card('What it noticed in this run', empty('This run has no finished attempts yet.')))
          else {
            const nums = spec.numbers ? await spec.numbers(c) : null
            if (nums && nums.length) parts.push(stats(nums))
            parts.push(await noticed(c))
            if (spec.extra) parts.push(...[await spec.extra(c)].flat())
          }
        } catch (err) { parts.push(card('What it noticed in this run', empty(`Could not read this run: ${err.message}`))) }
      }
      parts.push(settings(name))
      root.replaceChildren(...parts.filter(Boolean))
      const here = document.querySelector('.dk-attempt.dk-here')
      if (here && !scrolled) { here.scrollIntoView({ block: 'start' }); scrolled = true }
      return Boolean(run && run.status === 'running')
    }
    const tick = async () => { if (await render()) setTimeout(tick, 4000) }
    tick()
  }

  window.dk = { h, s, media, json, fmt, pct, plural, ep3, card, empty, row, stats, bars, hbars, table, downloadButton,
    attemptBlock, plugin, RUN, ATTEMPT, INFO }
})()
