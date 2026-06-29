
function initUIFailsafes() {
  const sidebarBtn = document.getElementById('sidebar-toggle');
  if (sidebarBtn && !sidebarBtn.dataset.failsafeBound) {
    sidebarBtn.dataset.failsafeBound = '1';
    sidebarBtn.addEventListener('click', () => {
      if (window.innerWidth < 960) return;
      const collapsed = !document.body.classList.contains('sidebar-collapsed');
      document.body.classList.toggle('sidebar-collapsed', collapsed);
      try { localStorage.setItem('signalforge_sidebar_collapsed', collapsed ? '1' : '0'); } catch (_) {}
      sidebarBtn.setAttribute('aria-pressed', collapsed ? 'true' : 'false');
    });
  }

  const viewSelect = document.querySelector('[data-view-mode-select="true"]');
  if (viewSelect && !viewSelect.dataset.failsafeBound) {
    viewSelect.dataset.failsafeBound = '1';
    viewSelect.addEventListener('change', () => {
      const mode = viewSelect.value === 'human' ? 'human' : 'analyst';
      document.documentElement.setAttribute('data-view-mode', mode);
      try { localStorage.setItem('forecast-view-mode', mode); } catch (_) {}
      if (typeof applyViewMode === 'function') applyViewMode(mode);
    });
  }
}

function initSidebarCurrentPage() {
  const path = window.location.pathname || '/';
  document.querySelectorAll('.sidebar-link').forEach((link) => {
    const href = link.getAttribute('href') || '';
    const isDashboard = href === '/' && path === '/';
    const isProjectNew = href === '/projects/new' && path.startsWith('/projects/new');
    const isRuns = href === '/runs' && (path === '/runs' || path.startsWith('/runs/compare'));
    const isCurrent = isDashboard || isProjectNew || isRuns;
    if (isCurrent) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
}


function initSidebarCenterpiece() {
  const toggleBtn = document.getElementById('sidebar-toggle');
  const brandCard = document.getElementById('brand-centerpiece');
  if (!toggleBtn) return;

  const storageKey = 'signalforge_sidebar_collapsed';
  const isCompactScreen = () => window.innerWidth < 960;

  function applyState(collapsed, persist = true) {
    const shouldCollapse = Boolean(collapsed) && !isCompactScreen();
    document.body.classList.toggle('sidebar-collapsed', shouldCollapse);
    toggleBtn.setAttribute('aria-pressed', shouldCollapse ? 'true' : 'false');
    toggleBtn.setAttribute('aria-label', shouldCollapse ? 'Expand sidebar' : 'Collapse sidebar');
    const icon = toggleBtn.querySelector('.sidebar-toggle__icon');
    if (icon) icon.textContent = shouldCollapse ? '⟩⟩' : '⟨⟨';
    if (persist) {
      try { localStorage.setItem(storageKey, shouldCollapse ? '1' : '0'); } catch (_) {}
    }
  }

  let saved = '0';
  try { saved = localStorage.getItem(storageKey) || '0'; } catch (_) {}
  applyState(saved === '1', false);

  toggleBtn.addEventListener('click', () => {
    applyState(!document.body.classList.contains('sidebar-collapsed'));
  });

  if (brandCard) {
    brandCard.addEventListener('dblclick', () => {
      applyState(!document.body.classList.contains('sidebar-collapsed'));
    });
    brandCard.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        applyState(!document.body.classList.contains('sidebar-collapsed'));
      }
    });
  }

  window.addEventListener('resize', () => {
    if (isCompactScreen()) {
      document.body.classList.remove('sidebar-collapsed');
      toggleBtn.setAttribute('aria-pressed', 'false');
      const icon = toggleBtn.querySelector('.sidebar-toggle__icon');
      if (icon) icon.textContent = '⟨⟨';
    } else {
      let latest = '0';
      try { latest = localStorage.getItem(storageKey) || '0'; } catch (_) {}
      applyState(latest === '1', false);
    }
  });
}


const TOOLTIP_LIBRARY = {
  visualization_controls: {title:'Visualization Controls', body:'Use these controls to reduce noise before you interpret the charts. Filtering time windows often makes trend shifts easier to spot.', tip:'Start with the last 90 or 180 days when you are diagnosing recent changes.'},
  run_summary: {title:'Run Summary', body:'This is the fastest way to confirm what the engine actually ran, including GPU usage, forecast start, and key diagnostics.', tip:'Use this block first when a result looks surprising.'},
  forecast_diagnostics: {title:'Forecast Diagnostics Time Frame', body:'Lead and cost diagnostics are calculated from the full aligned historical window shown on the page. Recent revenue diagnostics use the last 30 historical periods. Labels like per day, per week, or per month reflect the modeled period basis.', tip:'Always read the history window and period basis before comparing max, average, or recent diagnostic values.'},
  top_ranked_models: {title:'Top Ranked Models', body:'These are the models that performed best under your chosen ranking strategy. High-ranked models are usually the safest starting point for trust.', tip:'If the top scores are close together, the forecast is less dependent on a single model.'},
  chart_gallery: {title:'Chart Gallery', body:'This is where the forecast becomes visual. Use it to compare recent history, scenario bands, warnings, and cohort behavior.', tip:'Hide extra date history first, then compare direction and volatility.'},
  forecast_table: {title:'Forecast Table', body:'This is the auditable view of the output. It helps you inspect exact predicted values, intervals, and model assignments.', tip:'Use the search box to isolate a date or segment before exporting.'},
  early_warning_system: {title:'Early Warning System', body:'Warnings surface when the app detects pressure in revenue, leads, ROAS, or efficiency. They are meant to prioritize investigation, not replace judgment.', tip:'Focus on repeated warnings with higher severity before reacting to one-off spikes.'},
  revenue_waterfall: {title:'Revenue Change Waterfall', body:'The waterfall separates change into major components like lead volume, monetization efficiency, and mix. It turns the forecast into a business story.', tip:'Use this when stakeholders ask why revenue moved, not just how much it moved.'},
  driver_sensitivity: {title:'Driver Sensitivity', body:'This estimates how strongly external drivers are associated with revenue movement. The cleanest drivers are usually leads, spend, and major mapped regressors.', tip:'Treat very low-confidence relationships as directional hints, not hard rules.'},
  sheet_influence: {title:'Uploaded Sheet Influence', body:'This rolls driver effects up to the uploaded sheet level so you can see which dataset is materially shaping the forecast.', tip:'If a sheet has no influence, it may not be mapped cleanly or it may not add signal.'},
  goal_optimizer: {title:'Goal Optimizer', body:'This estimates what it takes to hit a target using historical budget, lead, ROAS, and monetization behavior. It is adaptive, not just simple arithmetic.', tip:'Compare balanced, efficient, and aggressive options before committing to a plan.'},
  scenario_engine: {title:'Scenario Engine', body:'This is your fast what-if sandbox. It lets you pressure-test assumptions like volume, spend, conversion quality, and user value.', tip:'Move one control at a time first so you can see which lever matters most.'},
  planning_workspace: {title:'Planning Workspace', body:'This is the premium planning studio for target setting, budget caps, ROAS pressure, and saved scenarios.', tip:'A plan can look exciting and still be unrealistic, so always check feasibility and confidence together.'},
  cohort_lens: {title:'Cohort Lens', body:'The cohort lens tracks how lead-month cohorts pay back over time and which platforms keep producing revenue after acquisition.', tip:'Use cohorts to learn and platforms to act.'},
  mapping_page: {title:'Mapping matters', body:'Most forecast problems begin here. Correct dates, target fields, and helper mappings matter more than model choice.', tip:'If something looks wrong later, revisit mapping first.'},
  upload_page: {title:'Upload strategy', body:'The core revenue file anchors the forecast. Helper files only help when they add trustworthy context like leads, spend, events, or cohorts.', tip:'Do not upload extra sheets just because they exist. Upload them because they explain revenue.'},
  setup_page: {title:'Setup strategy', body:'This page controls horizon, ranking logic, models, and GPU preference. Safer settings are easier to trust; aggressive settings explore more possibilities.', tip:'Balanced ranking is usually the best default.'},
  quality_page: {title:'Quality checks', body:'This page protects you from silent forecasting failures caused by missing dates, broken joins, or helper data that does not align cleanly.', tip:'A clean table beats a fancy model every time.'},
  home_page: {title:'Start flow', body:'The app is designed to move from project creation to upload, mapping, quality, setup, and finally results and planning.', tip:'Think of each page as reducing uncertainty before the forecast runs.'}
};

const PAGE_TIP_GROUPS = {
  general: ['home_page','upload_page','mapping_page'],
  home: ['home_page','upload_page','mapping_page'],
  project_new: ['home_page','upload_page','mapping_page'],
  upload: ['upload_page','mapping_page','quality_page'],
  mapping: ['mapping_page','quality_page','setup_page'],
  forecast_setup: ['setup_page','quality_page','run_summary'],
  quality: ['quality_page','mapping_page','setup_page'],
  results: ['run_summary','chart_gallery','forecast_table'],
  running: ['run_summary']
};

const HUMAN_MODE_GLOSSARY = [
  { key:'zscore', title:'Z-Score', humanTitle:'How unusual this is', analyst:'Shows how far a value is from normal in standard deviations. Around 0 is normal, +2 is unusually high, and -2 is unusually low.', human:'Tells you whether a number looks normal or unusual compared with the usual pattern. Bigger positive numbers mean an unusually high result; bigger negative numbers mean an unusually low one.' },
  { key:'roas', title:'ROAS', humanTitle:'Return on ad spend', analyst:'Revenue divided by advertising cost. A 2.0x ROAS means $2 of revenue for every $1 spent.', human:'Shows how much revenue came back for each $1 of ad spend. Example: 2.0x means every $1 spent brought back $2 in revenue.' },
  { key:'cohort', title:'Cohort', humanTitle:'Customer group from the same starting period', analyst:'A cohort is a group of users acquired in the same time window so payback and retention can be compared consistently over time.', human:'A cohort is simply a group of customers that started around the same time, such as people acquired in one month. This helps you see whether newer groups behave better or worse than older ones.' },
  { key:'feature_importance', title:'Feature Importance', humanTitle:'Main drivers behind the forecast', analyst:'Ranks which model inputs had the strongest influence on the selected forecast.', human:'Shows which business factors mattered most when the app built the forecast, such as leads, spend, seasonality, or quality changes.' },
  { key:'attribution', title:'Attribution', humanTitle:'Which channels deserve credit', analyst:'Directional channel scoring using the mapped revenue, cost, leads, and stability measures available in the run.', human:'Helps estimate which channels, affiliates, or platforms are pulling the most weight so you can see where revenue is likely coming from.' },
  { key:'anomaly', title:'Anomaly', humanTitle:'Possible abnormal behavior', analyst:"A value that deviates materially from historical expectation based on the app's anomaly logic.", human:'Flags a day or period that looks unusually high or low compared with normal behavior, so you know where to investigate first.' },
  { key:'confidence', title:'Confidence Score', humanTitle:'How stable this prediction looks', analyst:'A summary of how stable the forecast appears based on agreement, variance, and signal quality.', human:'Tells you how steady and believable the prediction looks. Higher confidence means the pattern is more consistent and less noisy.' },
  { key:'whale', title:'User', humanTitle:'Top-spending users', analyst:'High-value users that contribute a disproportionate share of revenue.', human:'These are the biggest spenders. A small number of users can drive a large share of total revenue, so changes here matter a lot.' }
];

const HUMAN_MODE_TEXT_SWAPS = [
  ['Forecast Avg / Period', 'Expected Average Per Period'],
  ['Top Ranked Models', 'Best-Performing Models'],
  ['Explainable AI', 'Why the Forecast Moved'],
  ['Channel Attribution', 'Channel Contribution'],
  ['Cohort Intelligence', 'Group Performance Over Time'],
  ['Cohorts Report', 'Cohort Summary'],
  ['Auto Alerts', 'Automatic Warnings'],
  ['Confidence Score', 'How Stable This Prediction Looks'],
  ['Feature Importance Summary', 'Main Drivers Behind the Forecast'],
  ['Lead-Month Cohort Revenue Lens', 'How Customer Groups Pay Back Over Time'],
  ['Anomaly Detection & Auto Alerts', 'Abnormal Trend Detection'],
  ['Scenario Simulator 2.0', 'What-If Planning'],
  ['Goal Optimizer', 'Target Planner'],
  ['Revenue Change Waterfall', 'What Changed Revenue'],
  ['Driver Sensitivity', 'What Revenue Responds To'],
  ['Recent Cohort ROAS', 'Recent Cohort Return on Ad Spend'],
  ['Cohort Payback Summary', 'Cohort Payback Summary'],
  ['User Intelligence 2.0', 'High-Value User Intelligence']
];

function tooltipHtml(entry) {
  if (!entry) return '';
  const title = entry.title ? `<strong>${entry.title}</strong><br>` : '';
  const body = entry.body || '';
  const tip = entry.tip ? `<br><br><em>Tip:</em> ${entry.tip}` : '';
  return `${title}${body}${tip}`;
}


function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function getTooltipEntryForNode(node) {
  const key = node.getAttribute('data-tooltip-key');
  if (key && TOOLTIP_LIBRARY[key]) return TOOLTIP_LIBRARY[key];
  const body = node.getAttribute('data-tooltip');
  if (!body) return null;
  return { title: '', body, tip: '' };
}

function buildTooltipCard(entry) {
  if (!entry) return '';
  const title = entry.title ? `<div class="hovercard-title">${escapeHtml(entry.title)}</div>` : '';
  const body = entry.body ? `<div class="hovercard-body">${escapeHtml(entry.body)}</div>` : '';
  const tip = entry.tip ? `<div class="hovercard-tip"><span>Tip</span>${escapeHtml(entry.tip)}</div>` : '';
  return `${title}${body}${tip}`;
}


function bindFallbackTooltips() {
  document.querySelectorAll('.help-tip[data-tooltip]').forEach((node) => {
    if (node.dataset.tooltipBound === '1') return;
    node.dataset.tooltipBound = '1';

    const showNative = () => {
      const text = node.getAttribute('data-tooltip') || node.getAttribute('title') || '';
      if (text && !node.getAttribute('title')) node.setAttribute('title', text);
    };

    node.addEventListener('mouseenter', showNative);
    node.addEventListener('focus', showNative);
  });
}

function initHoverCards() {
  const nodes = Array.from(document.querySelectorAll('.help-tip[data-tooltip], .help-tip[data-tooltip-key]'));
  if (!nodes.length) return;

  document.documentElement.classList.add('hovercards-enabled');

  document.querySelectorAll('.smart-hovercard').forEach((node) => node.remove());

  let activeNode = null;
  let hideTimer = null;
  const card = document.createElement('div');
  card.className = 'smart-hovercard';
  card.setAttribute('role', 'tooltip');
  card.setAttribute('aria-hidden', 'true');
  card.innerHTML = '<div class="smart-hovercard-arrow"></div><div class="smart-hovercard-inner"></div>';
  document.body.appendChild(card);
  const inner = card.querySelector('.smart-hovercard-inner');
  const arrow = card.querySelector('.smart-hovercard-arrow');

  function clearHide() {
    if (hideTimer) {
      window.clearTimeout(hideTimer);
      hideTimer = null;
    }
  }

  function placeCard(node) {
    const rect = node.getBoundingClientRect();
    const cardRect = card.getBoundingClientRect();
    const margin = 14;
    const gap = 12;
    let left = rect.left + (rect.width / 2) - (cardRect.width / 2);
    let top = rect.bottom + gap;
    let placement = 'bottom';

    if (top + cardRect.height > window.innerHeight - margin) {
      const aboveTop = rect.top - cardRect.height - gap;
      if (aboveTop >= margin) {
        top = aboveTop;
        placement = 'top';
      }
    }

    if (left < margin) left = margin;
    if (left + cardRect.width > window.innerWidth - margin) {
      left = window.innerWidth - margin - cardRect.width;
    }

    if (placement === 'bottom' && top + cardRect.height > window.innerHeight - margin) {
      top = Math.max(margin, window.innerHeight - margin - cardRect.height);
    }
    if (placement === 'top' && top < margin) {
      top = margin;
    }

    const anchorCenter = rect.left + rect.width / 2;
    let arrowLeft = anchorCenter - left - 8;
    arrowLeft = Math.max(14, Math.min(cardRect.width - 22, arrowLeft));

    card.dataset.placement = placement;
    card.style.left = `${Math.round(left)}px`;
    card.style.top = `${Math.round(top)}px`;
    arrow.style.left = `${Math.round(arrowLeft)}px`;
  }

  function showCard(node) {
    clearHide();
    const entry = getTooltipEntryForNode(node);
    if (!entry) return;
    activeNode = node;
    inner.innerHTML = buildTooltipCard(entry);
    card.classList.add('is-visible');
    card.setAttribute('aria-hidden', 'false');
    node.setAttribute('aria-expanded', 'true');
    requestAnimationFrame(() => placeCard(node));
  }

  function hideCard(immediate = false) {
    clearHide();
    const perform = () => {
      if (activeNode) activeNode.setAttribute('aria-expanded', 'false');
      activeNode = null;
      card.classList.remove('is-visible');
      card.setAttribute('aria-hidden', 'true');
    };
    if (immediate) {
      perform();
    } else {
      hideTimer = window.setTimeout(perform, 90);
    }
  }

  nodes.forEach((node, index) => {
    if (!node.id) node.id = `help-tip-${index + 1}`;
    node.setAttribute('tabindex', '0');
    node.setAttribute('aria-haspopup', 'true');
    node.setAttribute('aria-expanded', 'false');
    node.setAttribute('aria-describedby', 'smart-hovercard');
    node.addEventListener('mouseenter', () => showCard(node));
    node.addEventListener('focus', () => showCard(node));
    node.addEventListener('mouseleave', () => hideCard());
    node.addEventListener('blur', () => hideCard());
    node.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (activeNode === node && card.classList.contains('is-visible')) hideCard(true);
      else showCard(node);
    });
    node.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') hideCard(true);
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        if (activeNode === node && card.classList.contains('is-visible')) hideCard(true);
        else showCard(node);
      }
    });
  });

  card.id = 'smart-hovercard';
  card.addEventListener('mouseenter', clearHide);
  card.addEventListener('mouseleave', () => hideCard());
  window.addEventListener('scroll', () => {
    if (activeNode && card.classList.contains('is-visible')) placeCard(activeNode);
  }, true);
  window.addEventListener('resize', () => {
    if (activeNode && card.classList.contains('is-visible')) placeCard(activeNode);
  });
  document.addEventListener('click', (event) => {
    if (!card.contains(event.target) && !event.target.closest('.help-tip')) hideCard(true);
  });
}



function forceShowWorkspaceTips() {
  const rail = document.getElementById('page-tip-rail');
  const globalHelp = document.querySelector('.global-help');
  document.body.classList.remove('workspace-tips-hidden');
  if (globalHelp) globalHelp.style.display = '';
  if (rail) {
    rail.style.display = 'grid';
    rail.hidden = false;
    rail.dataset.loaded = '';
    if (typeof rebuildPageTipRail === 'function') rebuildPageTipRail(true);
    window.setTimeout(() => {
      rail.style.display = 'grid';
      rail.hidden = false;
      rail.querySelectorAll('.context-tip-card').forEach(card => {
        card.style.display = 'block';
        card.hidden = false;
      });
    }, 40);
  }
}

function forceHideWorkspaceTips() {
  document.body.classList.add('workspace-tips-hidden');
}
function rebuildPageTipRail(force = false) {
  const rail = document.getElementById('page-tip-rail');
  const globalHelp = document.querySelector('.global-help');
  if (!rail) return;
  if (force) {
    rail.dataset.loaded = '';
    rail.innerHTML = '';
  }
  const group = rail.getAttribute('data-tip-group') || 'general';
  const keys = PAGE_TIP_GROUPS[group] || PAGE_TIP_GROUPS.general;
  if (!keys || !keys.length) {
    rail.style.display = 'none';
    if (globalHelp) globalHelp.style.display = 'none';
    rail.dataset.loaded = 'true';
    return;
  }
  rail.style.display = '';
  if (globalHelp) globalHelp.style.display = '';
  rail.innerHTML = keys.map((key) => {
    const entry = TOOLTIP_LIBRARY[key];
    if (!entry) return '';
    return `<section class="context-tip-card is-dynamic"><div class="eyebrow">Quick tip</div><strong>${entry.title}</strong><p>${entry.body}</p>${entry.tip ? `<div class="tip-kicker">💡 ${entry.tip}</div>` : ''}</section>`;
  }).join('');
  rail.dataset.loaded = 'true';
}

function initDynamicTooltips() {
  document.querySelectorAll('.help-tip[data-tooltip-key], .help-tip[data-tooltip]').forEach((node, index) => {
    const entry = getTooltipEntryForNode(node);
    if (!entry) return;
    const plain = `${entry.title ? entry.title + '. ' : ''}${entry.body || ''}${entry.tip ? ' Tip: ' + entry.tip : ''}`.trim();

    node.setAttribute('data-tooltip', plain);
    node.setAttribute('title', plain);
    node.setAttribute('aria-label', plain);
    node.setAttribute('tabindex', '0');
    if (!node.id) node.id = `help-tip-${index + 1}`;
  });

  document.querySelectorAll('[data-inline-tip-key]').forEach((node) => {
    const entry = TOOLTIP_LIBRARY[node.getAttribute('data-inline-tip-key')];
    if (!entry) return;
    if (node.querySelector('.inline-smart-tip')) return;
    const box = document.createElement('div');
    box.className = 'inline-smart-tip';
    box.innerHTML = `<strong>${entry.title}.</strong> ${entry.tip || entry.body}`;
    node.appendChild(box);
  });

  const rail = document.getElementById('page-tip-rail');
  if (rail && !rail.dataset.loaded) {
    rebuildPageTipRail(true);
  }

  if (typeof bindFallbackTooltips === 'function') {
    bindFallbackTooltips();
  }
}


const THEME_CSS_MAP = {
  'arctic-glass': 'arctic-glass',
  'midnight-blue': 'midnight-blue',
  dark: 'dark',
  light: 'light',
  'chaos-black': 'chaos-black',
  stellar: 'stellar',
  forest: 'forest',
  sunset: 'sunset',
  'pastel-sky': 'pastel-sky',
  'pastel-lavender': 'pastel-lavender',
  'pastel-mint': 'pastel-mint',
  'pastel-peach': 'pastel-peach',
  'cobalt-grid': 'cobalt-grid',
  'emerald-grid': 'emerald-grid'
};

const AVAILABLE_THEMES = new Set(Object.keys(THEME_CSS_MAP));
const AVAILABLE_CHART_PALETTES = new Set(['balanced', 'aurora', 'sunset-pop', 'ocean-glow', 'orchid-punch', 'emerald-flow']);

function normalizeThemeName(theme) {
  return AVAILABLE_THEMES.has(theme) ? theme : 'midnight-blue';
}

function resolveThemeCssName(theme) {
  const normalized = normalizeThemeName(theme);
  return THEME_CSS_MAP[normalized] || 'midnight-blue';
}

function normalizeChartPaletteName(palette) {
  return AVAILABLE_CHART_PALETTES.has(palette) ? palette : 'balanced';
}

function getCurrentChartPalette() {
  const palette = document.documentElement.getAttribute('data-chart-palette') || 'balanced';
  return normalizeChartPaletteName(palette);
}

function getChartPaletteTokens(name, isDark) {
  const palettes = {
    balanced: {
      colorway: isDark
        ? ['#ff7a59', '#149b90', '#7ad4b0', '#ffb347', '#7a6ff0', '#3e7bff', '#e287b8', '#96a7d8']
        : ['#ff7f62', '#1a9f93', '#84d8b6', '#ffb44d', '#7d73f2', '#4380ff', '#de8fb5', '#9aabda'],
      semantic: { cost: '#ff7a59', leads: '#24c8f2', revenue: '#4f88ff', revenueAlt: '#6f63f6', loss: '#ff7a59', warning: '#ffbf54', warningDeep: '#c96a1c' }
    },
    aurora: {
      colorway: isDark
        ? ['#7ef0d2', '#67b7ff', '#8f86ff', '#f58bd7', '#ffd166', '#3ed1a1', '#5f89ff', '#d6a3ff']
        : ['#48d9b7', '#499eff', '#7c72ff', '#e47cc3', '#f4bc44', '#2fbe92', '#5f83ff', '#c98def'],
      semantic: { cost: '#ff8a66', leads: '#1ecfe8', revenue: '#5b95ff', revenueAlt: '#8b78ff', loss: '#ff8a66', warning: '#ffd166', warningDeep: '#da8b1d' }
    },
    'sunset-pop': {
      colorway: isDark
        ? ['#ff8a5b', '#ffb84d', '#ff6fa8', '#8b7cff', '#59d7c2', '#ffd98c', '#ff7f7f', '#7db8ff']
        : ['#ff8657', '#ffb03e', '#f56aa0', '#8173ff', '#49cab4', '#f7cf78', '#f67676', '#73aefe'],
      semantic: { cost: '#ff8a5b', leads: '#20cdb5', revenue: '#8b7cff', revenueAlt: '#ff6fa8', loss: '#ff7b6b', warning: '#ffbf57', warningDeep: '#d86f1c' }
    },
    'ocean-glow': {
      colorway: isDark
        ? ['#55d6ff', '#2fb7a8', '#5a8dff', '#7ee6d8', '#9f9cff', '#ffd166', '#4fc3ff', '#88b7ff']
        : ['#42c8f5', '#23a999', '#4d7df5', '#6adcca', '#8d88ff', '#f0be4e', '#42b8f0', '#79a8f7'],
      semantic: { cost: '#ff8a66', leads: '#18cfe9', revenue: '#4f88ff', revenueAlt: '#2fb7a8', loss: '#ff8266', warning: '#ffd166', warningDeep: '#cb7a1e' }
    },
    'orchid-punch': {
      colorway: isDark
        ? ['#a884ff', '#f38ad8', '#5f8fff', '#ff9b6b', '#6fe4d6', '#d7a6ff', '#ffd166', '#8eb4ff']
        : ['#9877ff', '#ea7ccc', '#5683ff', '#ff9160', '#5fd6c8', '#c997ff', '#f2c85b', '#84aaf9'],
      semantic: { cost: '#ff9468', leads: '#33d6c9', revenue: '#5f8fff', revenueAlt: '#a884ff', loss: '#ff7f73', warning: '#ffd166', warningDeep: '#c86a2f' }
    },
    'emerald-flow': {
      colorway: isDark
        ? ['#2fc77a', '#4c87ff', '#ff9f43', '#7b61ff', '#18cfe9', '#e9528c', '#ffd166', '#6f7f93']
        : ['#22b66e', '#3b74f2', '#f28c2f', '#7057f5', '#12bcd4', '#db4d82', '#e7b93f', '#74849a'],
      semantic: { cost: '#f28c2f', leads: '#12bcd4', revenue: '#3b74f2', revenueAlt: '#7057f5', loss: '#e05252', warning: '#e7b93f', warningDeep: '#9a5c00' }
    }
  };
  return palettes[name] || palettes.balanced;
}

function applyTheme(theme, options = {}) {
  const normalized = normalizeThemeName(theme);
  const previous = document.documentElement.getAttribute('data-theme') || 'midnight-blue';
  const silent = options && options.silent === true;
  const resolvedTheme = resolveThemeCssName(normalized);
  const currentSource = document.documentElement.getAttribute('data-theme-source') || previous;
  if (previous === resolvedTheme && currentSource === normalized && !options.force) {
    const select = document.querySelector('[data-theme-select="true"]');
    if (select && select.value !== normalized) select.value = normalized;
    return;
  }
  document.documentElement.classList.add('theme-is-switching');
  if (!silent) {
    document.documentElement.setAttribute('data-theme-notice', 'Applying theme...');
  }
  document.documentElement.setAttribute('data-theme', resolvedTheme);
  document.documentElement.setAttribute('data-theme-source', normalized);
  const select = document.querySelector('[data-theme-select="true"]');
  if (select && select.value !== normalized) select.value = normalized;
  try { localStorage.setItem('forecast-theme', normalized); } catch (error) {}
  window.dispatchEvent(new CustomEvent('forecast-theme-change', {
    detail: { theme: normalized, previousTheme: previous }
  }));
  window.setTimeout(() => {
    if (typeof forceChartThemeSync === 'function') forceChartThemeSync();
  }, 60);
  window.setTimeout(() => {
    if (typeof forceChartThemeSync === 'function') forceChartThemeSync();
  }, 220);
  window.setTimeout(() => {
    if (typeof forceChartThemeSync === 'function') forceChartThemeSync();
  }, 520);
  window.setTimeout(() => {
    document.documentElement.classList.remove('theme-is-switching');
  }, 260);
  if (window.__sfThemeLoaderTimer) {
    window.clearTimeout(window.__sfThemeLoaderTimer);
    window.__sfThemeLoaderTimer = null;
  }
  if (!silent) {
    window.__sfThemeLoaderTimer = window.setTimeout(() => {
      document.documentElement.classList.remove('theme-is-switching');
      document.documentElement.removeAttribute('data-theme-notice');
      window.__sfThemeLoaderTimer = null;
    }, 640);
  }
}



function getPlotlyThemeTokens() {
  const styles = getComputedStyle(document.documentElement);
  const theme = document.documentElement.getAttribute('data-theme') || 'arctic-glass';
  const darkThemes = new Set(['midnight-blue', 'dark', 'chaos-black', 'stellar', 'forest', 'sunset']);
  const isDark = darkThemes.has(theme);

  const rawText = (styles.getPropertyValue('--text') || '').trim();
  const rawMuted = (styles.getPropertyValue('--muted') || '').trim();
  const rawBorder = (styles.getPropertyValue('--border') || '').trim();
  const rawPanel = (styles.getPropertyValue('--panel') || '').trim();
  const rawPanel2 = (styles.getPropertyValue('--panel-2') || '').trim();
  const rawSurfaceStrong = (styles.getPropertyValue('--surface-strong') || '').trim();
  const rawSurfaceInput = (styles.getPropertyValue('--surface-input') || '').trim();
  const rawAccent = (styles.getPropertyValue('--accent') || '').trim();
  const rawAccent2 = (styles.getPropertyValue('--accent-2') || '').trim();
  const paletteName = getCurrentChartPalette();
  const paletteTokens = getChartPaletteTokens(paletteName, isDark);
  const neutralPaper = rawPanel || (isDark ? '#101826' : '#ffffff');
  const neutralPlot = isDark
    ? (rawPanel2 || rawSurfaceInput || rawPanel || '#172132')
    : (rawSurfaceInput || rawPanel || '#ffffff');

  return {
    theme,
    chartPalette: paletteName,
    isDark,
    text: rawText || (isDark ? '#E6EDF3' : '#1A1F2B'),
    muted: rawMuted || (isDark ? '#9FB0C8' : '#5B6B82'),
    border: rawBorder || (isDark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.10)'),
    grid: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)',
    zero: isDark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.14)',
    panel: rawPanel || (isDark ? '#101826' : '#ffffff'),
    panel2: rawPanel2 || rawSurfaceStrong || rawPanel || (isDark ? '#172132' : '#f7f9fc'),
    paper: neutralPaper,
    plot: neutralPlot,
    accent: rawAccent || '#5d63f0',
    accent2: rawAccent2 || '#0f9d76',
    thresholds: isDark
      ? {
          negative: '#ff7a7a',
          positive: '#2bb534',
          info: '#75cfff',
          neutral: '#a4a4ad'
        }
      : {
          negative: '#db0000',
          positive: '#00802f',
          info: '#006ce0',
          neutral: '#656871'
        },
    colorway: paletteTokens.colorway,
    semantic: paletteTokens.semantic,
    watermark: isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)'
  };
}



function applyPlotlyThemeToElement(el) {
  if (!el || !window.Plotly) return;
  const themeSource = el._plotlyThemeSource || el._plotlyOriginal || { data: el.data, layout: el.layout };
  const sourceData = Array.isArray(themeSource.data) ? themeSource.data : (el.data || []);
  const sourceLayout = themeSource.layout || el.layout || {};
  if (!sourceData.length || !sourceLayout) return;
  const t = getPlotlyThemeTokens();
  const semanticColors = t.semantic || {};
  const thresholdColors = t.thresholds || {};
  const pickThresholdColor = (value, fallback) => {
    const label = String(value || '').toLowerCase();
    if (!label) return fallback;
    if (label.includes('negative') || label.includes('max') || label.includes('maximum') || label.includes('limit') || label.includes('loss') || label.includes('fail')) return thresholdColors.negative || fallback;
    if (label.includes('positive') || label.includes('pass') || label.includes('goal') || label.includes('target') || label.includes('success')) return thresholdColors.positive || fallback;
    if (label.includes('info') || label.includes('forecast') || label.includes('estimate') || label.includes('projected')) return thresholdColors.info || fallback;
    if (label.includes('neutral') || label.includes('average') || label.includes('baseline') || label.includes('benchmark')) return thresholdColors.neutral || fallback;
    return fallback;
  };
  const pickSemanticColor = (trace, fallback) => {
    const label = `${trace?.name || ''} ${trace?.legendgroup || ''}`.toLowerCase();
    if (label.includes('cost') || label.includes('spend')) return semanticColors.cost;
    if (label.includes('lead')) return semanticColors.leads;
    if (label.includes('revenue')) return semanticColors.revenue;
    if (label.includes('loss') || label.includes('downside') || label.includes('declin') || label.includes('drop')) return semanticColors.loss;
    if (label.includes('roas') || label.includes('upside') || label.includes('target') || label.includes('forecast')) return semanticColors.warning;
    if (label.includes('risk') || label.includes('pressure')) return semanticColors.warningDeep;
    return fallback;
  };
  const isDateLikeValue = (value) => {
    if (value instanceof Date) return true;
    if (typeof value !== 'string') return false;
    const trimmed = value.trim();
    if (!trimmed) return false;
    if (/^\d{4}-\d{1,2}-\d{1,2}/.test(trimmed)) return true;
    if (/^\d{1,2}\/\d{1,2}\/\d{2,4}$/.test(trimmed)) return true;
    const parsed = Date.parse(trimmed);
    return Number.isFinite(parsed) && /[a-zA-Z]/.test(trimmed);
  };
  const isNumericLikeValue = (value) => {
    if (typeof value === 'number') return Number.isFinite(value);
    if (typeof value !== 'string') return false;
    const trimmed = value.trim();
    if (!trimmed) return false;
    return Number.isFinite(Number(trimmed));
  };
  const getCategoryValues = (trace) => Array.isArray(trace?.x) && trace.x.length ? trace.x : (Array.isArray(trace?.y) ? trace.y : []);
  const hasCategoricalBarAxis = (trace) => {
    const values = getCategoryValues(trace);
    if (!values.length) return false;
    return values.some((value) => !isNumericLikeValue(value) && !isDateLikeValue(value));
  };
  const getLongestLabelLength = (trace) => getCategoryValues(trace).reduce((max, value) => Math.max(max, String(value ?? '').length), 0);
  const singleBarTrace = sourceData.filter((trace) => trace?.type === 'bar' || trace?.type === 'histogram');
  const hasSingleCategoricalBarTrace = singleBarTrace.length === 1 && hasCategoricalBarAxis(singleBarTrace[0]);
  const longestCategoricalLabel = hasSingleCategoricalBarTrace ? getLongestLabelLength(singleBarTrace[0]) : 0;

  const titleObj = (sourceLayout.title && typeof sourceLayout.title === 'object') ? sourceLayout.title : {};
  const legendObj = sourceLayout.legend || {};
  const xaxisObj = sourceLayout.xaxis || {};
  const yaxisObj = sourceLayout.yaxis || {};
  const yaxis2Obj = sourceLayout.yaxis2 || {};
  const layoutShapes = Array.isArray(sourceLayout.shapes) ? sourceLayout.shapes : [];
  const layoutAnnotations = Array.isArray(sourceLayout.annotations) ? sourceLayout.annotations : [];

  const nextLayout = {
    font: {
      family: 'Inter, Segoe UI, Roboto, Arial, sans-serif',
      size: 14,
      color: t.text
    },
    paper_bgcolor: t.paper,
    plot_bgcolor: t.plot,
    colorway: t.colorway,
    margin: {
      t: Math.max(72, Number((sourceLayout.margin || {}).t || (el.layout.margin || {}).t || 48)),
      l: Math.max(60, Number((el.layout.margin || {}).l || 44)),
      r: Math.max(44, Number((el.layout.margin || {}).r || 28)),
      b: Math.max(
        hasSingleCategoricalBarTrace ? (longestCategoricalLabel > 12 ? 128 : 108) : 92,
        Number((sourceLayout.margin || {}).b || (el.layout.margin || {}).b || 52)
      ),
      pad: Math.max(10, Number((el.layout.margin || {}).pad || 6))
    },
    title: titleObj && Object.keys(titleObj).length
      ? {
          ...titleObj,
          font: {
            ...(titleObj.font || {}),
            family: 'Inter, Segoe UI, Roboto, Arial, sans-serif',
            size: 18,
            color: t.text
          }
        }
      : titleObj,
    legend: {
      ...legendObj,
      font: { ...(legendObj.font || {}), size: 13, color: t.muted },
      bgcolor: 'rgba(0,0,0,0)'
    },
    xaxis: {
      ...xaxisObj,
      color: t.text,
      tickfont: { ...(xaxisObj.tickfont || {}), size: 13, color: t.muted },
      title: { ...(xaxisObj.title || {}), font: { ...(((xaxisObj.title || {}).font) || {}), size: 14, color: t.text } },
      gridcolor: t.grid,
      zerolinecolor: t.zero,
      linecolor: t.border,
      automargin: true,
      tickangle: hasSingleCategoricalBarTrace
        ? (Number.isFinite(Number(xaxisObj.tickangle)) ? xaxisObj.tickangle : (longestCategoricalLabel > 12 ? -28 : -12))
        : xaxisObj.tickangle,
      ticklabeloverflow: 'allow'
    },
    yaxis: {
      ...yaxisObj,
      color: t.text,
      tickfont: { ...(yaxisObj.tickfont || {}), size: 13, color: t.muted },
      title: { ...(yaxisObj.title || {}), font: { ...(((yaxisObj.title || {}).font) || {}), size: 14, color: t.text } },
      gridcolor: t.grid,
      zerolinecolor: t.zero,
      linecolor: t.border,
      automargin: true,
      ticklabeloverflow: 'allow'
    },
    yaxis2: {
      ...yaxis2Obj,
      color: t.text,
      tickfont: { ...(yaxis2Obj.tickfont || {}), size: 13, color: t.muted },
      title: { ...(yaxis2Obj.title || {}), font: { ...(((yaxis2Obj.title || {}).font) || {}), size: 14, color: t.text } },
      gridcolor: t.grid,
      zerolinecolor: t.zero,
      linecolor: t.border,
      automargin: true
    },
    hoverlabel: {
      ...(sourceLayout.hoverlabel || {}),
      bgcolor: t.isDark ? '#121826' : '#ffffff',
      bordercolor: t.border,
      font: { ...((sourceLayout.hoverlabel || {}).font || {}), color: t.text }
    },
    shapes: layoutShapes.map((shape) => {
      const label = `${shape?.name || ''} ${(shape?.label || {}).text || ''} ${shape?.templateitemname || ''}`;
      const shapeColor = pickThresholdColor(label, shape?.line?.color || t.border);
      return {
        ...shape,
        line: {
          ...(shape.line || {}),
          color: shapeColor
        }
      };
    }),
    annotations: layoutAnnotations.map((annotation) => {
      const annotationColor = pickThresholdColor(annotation?.text, annotation?.font?.color || t.text);
      return {
        ...annotation,
        font: {
          ...(annotation.font || {}),
          color: annotationColor
        },
        arrowcolor: pickThresholdColor(annotation?.text, annotation.arrowcolor || t.border)
      };
    })
  };

  const themedData = sourceData.map((trace, index) => {
    const next = { ...trace };
    const traceLabel = `${trace?.name || ''} ${trace?.legendgroup || ''}`.toLowerCase();
    const paletteColor = pickSemanticColor(trace, t.colorway[index % t.colorway.length]);
    const isSevenDayAverage = traceLabel.includes('7-day') || traceLabel.includes('7 day') || traceLabel.includes('7d');
    const isThirtyDayAverage = traceLabel.includes('30-day') || traceLabel.includes('30 day') || traceLabel.includes('30d');
    const isMovingAverage = (
      traceLabel.includes('avg') ||
      traceLabel.includes('average') ||
      traceLabel.includes('moving') ||
      traceLabel.includes('smooth')
    );
    const pointCount = Math.max(
      Array.isArray(trace?.x) ? trace.x.length : 0,
      Array.isArray(trace?.y) ? trace.y.length : 0
    );
    const shouldUseCategoryPalette = (
      trace.type === 'bar' &&
      hasSingleCategoricalBarTrace &&
      pointCount > 1 &&
      (!Array.isArray((trace.marker || {}).color)) &&
      typeof (trace.marker || {}).color !== 'object'
    );

    if (trace.type === 'waterfall') {
      next.cliponaxis = false;
      next.constraintext = 'none';
      next.textfont = { ...(trace.textfont || {}), color: t.text, size: 14 };
      next.insidetextfont = { ...(trace.insidetextfont || {}), color: t.text, size: 14 };
      next.outsidetextfont = { ...(trace.outsidetextfont || {}), color: t.text, size: 14 };
      next.connector = { ...(trace.connector || {}), line: { ...((trace.connector || {}).line || {}), color: t.border } };
      next.increasing = { ...(trace.increasing || {}), marker: { ...((trace.increasing || {}).marker || {}), color: semanticColors.revenue } };
      next.decreasing = { ...(trace.decreasing || {}), marker: { ...((trace.decreasing || {}).marker || {}), color: semanticColors.loss } };
      next.totals = { ...(trace.totals || {}), marker: { ...((trace.totals || {}).marker || {}), color: semanticColors.warning } };
    }

    if (trace.type === 'bar' || trace.type === 'histogram') {
      next.cliponaxis = false;
      next.constraintext = 'none';
      next.textfont = { ...(trace.textfont || {}), color: t.text, size: 14 };
      const categoryColors = shouldUseCategoryPalette
        ? Array.from({ length: pointCount }, (_, pointIndex) => t.colorway[pointIndex % t.colorway.length])
        : null;
      next.marker = {
        ...(trace.marker || {}),
        color: categoryColors || (trace.marker || {}).color || paletteColor,
        line: {
          color: shouldUseCategoryPalette
            ? 'rgba(255,255,255,0.34)'
            : (((trace.marker || {}).line || {}).color || t.border),
          width: shouldUseCategoryPalette
            ? 0.75
            : ((((trace.marker || {}).line) || {}).width || 0.5),
          ...(((trace.marker || {}).line) || {})
        }
      };
      if (traceLabel.includes('lead')) {
        next.opacity = Math.max(0.88, Number(trace.opacity || 0));
        next.marker = {
          ...(next.marker || trace.marker || {}),
          color: ((next.marker || trace.marker || {}).color) || semanticColors.leads,
          line: {
            color: semanticColors.leads,
            width: 1.1,
            ...((((next.marker || trace.marker || {}).line) || {}))
          }
        };
      }
    }

    if (trace.textfont || trace.textposition || trace.insidetextfont || trace.outsidetextfont) {
      next.textfont = { ...(trace.textfont || {}), color: t.text, size: 14 };
      next.insidetextfont = { ...(trace.insidetextfont || {}), color: t.text, size: 14 };
      next.outsidetextfont = { ...(trace.outsidetextfont || {}), color: t.text, size: 14 };
    }

    if (trace.type === 'scatter') {
      next.cliponaxis = false;
      next.line = {
        ...(trace.line || {}),
        color: (trace.line || {}).color || paletteColor,
        width: (trace.line && trace.line.width) || 2
      };
      next.marker = {
        ...(trace.marker || {}),
        color: (trace.marker || {}).color || paletteColor,
        line: { ...((trace.marker || {}).line || {}), color: t.border, width: ((trace.marker || {}).line || {}).width || 0.5 }
      };
      if (traceLabel.includes('revenue')) {
        next.line = {
          ...(next.line || {}),
          color: (trace.line || {}).color || semanticColors.revenue,
          width: Math.max(2.6, Number((trace.line || {}).width || 0))
        };
      }
      if (isSevenDayAverage) {
        next.line = {
          ...(next.line || {}),
          color: '#2ea88f',
          width: Math.max(2.4, Number((trace.line || {}).width || 0)),
          dash: 'solid'
        };
      } else if (isThirtyDayAverage) {
        next.line = {
          ...(next.line || {}),
          color: semanticColors.revenueAlt || '#7a63f6',
          width: Math.max(2.3, Number((trace.line || {}).width || 0)),
          dash: 'dot'
        };
      } else if (isMovingAverage && !traceLabel.includes('revenue') && !traceLabel.includes('lead')) {
        next.line = {
          ...(next.line || {}),
          color: semanticColors.revenueAlt || paletteColor,
          width: Math.max(2.2, Number((trace.line || {}).width || 0)),
          dash: 'solid'
        };
      }
      if (traceLabel.includes('lead')) {
        next.line = {
          ...(next.line || {}),
          color: (trace.line || {}).color || semanticColors.leads,
          width: Math.max(2.2, Number((trace.line || {}).width || 0))
        };
        next.marker = {
          ...(next.marker || {}),
          color: (trace.marker || {}).color || semanticColors.leads,
          opacity: Math.max(0.9, Number((trace.marker || {}).opacity || 0)),
          line: {
            ...((next.marker || {}).line || {}),
            color: semanticColors.leads,
            width: Math.max(0.8, Number((((trace.marker || {}).line) || {}).width || 0))
          }
        };
        if (!next.fillcolor && trace.fill) {
          next.fillcolor = t.isDark ? 'rgba(36, 200, 242, 0.28)' : 'rgba(36, 200, 242, 0.22)';
        }
      }
    }

    if (trace.mode && String(trace.mode).includes('text')) {
      next.textfont = { ...(trace.textfont || {}), color: t.text, size: 14 };
    }

    if (trace.name && String(trace.name).toLowerCase().includes('baseline')) {
      next.line = { ...(trace.line || {}), color: thresholdColors.neutral || semanticColors.revenueAlt, dash: 'dot' };
    }

    if (trace.name && String(trace.name).toLowerCase().includes('threshold')) {
      next.line = { ...(trace.line || {}), color: pickThresholdColor(trace.name, (trace.line || {}).color || thresholdColors.info || t.text), dash: (trace.line || {}).dash || 'dash' };
      next.marker = { ...(trace.marker || {}), color: pickThresholdColor(trace.name, (trace.marker || {}).color || thresholdColors.info || t.text) };
    }

    if (trace.name && String(trace.name).toLowerCase().includes('anomal')) {
      next.marker = { ...(trace.marker || {}), color: semanticColors.warning, line: { color: t.border, width: 0.5 } };
    }

    if ((trace.type === 'pie' || trace.type === 'funnelarea') && !trace.marker?.colors) {
      next.marker = {
        ...(trace.marker || {}),
        colors: t.colorway
      };
    }

    return next;
  });

  try {
    window.Plotly.react(el, themedData, nextLayout, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['select2d', 'lasso2d']
    });
    el.style.backgroundColor = t.paper;
    el.style.borderRadius = '14px';
    el.style.width = '100%';
    try {
      const textFontStyle = {color: t.text, size: 14, family: 'Inter, Segoe UI, Roboto, Arial, sans-serif'};
      window.Plotly.restyle(el, {
        textfont: [textFontStyle],
        insidetextfont: [textFontStyle],
        outsidetextfont: [textFontStyle],
        marker: themedData.map(trace => trace.marker || {})
      });
      window.Plotly.relayout(el, {
        'font.color': t.text,
        'legend.font.color': t.text,
        'xaxis.tickfont.color': t.text,
        'yaxis.tickfont.color': t.text,
        'xaxis.title.font.color': t.text,
        'yaxis.title.font.color': t.text
      });
    } catch (error) {}
    el._plotlyThemedSnapshot = { data: themedData, layout: nextLayout };
  } catch (error) {
    console.warn('Plotly theme relayout skipped for', el.id || el, error);
  }
}



function syncVisibleChartsToTheme() {
  if (!window.Plotly) return;
  document.querySelectorAll('.js-plotly-plot, .plotly-graph-div, .dashboard-chart, [data-plotly-spec]').forEach((el) => {
    try {
      if (typeof applyPlotlyThemeToElement === 'function') applyPlotlyThemeToElement(el);
      const box = el.getBoundingClientRect();
      if (!box.width || !box.height || el.offsetParent === null) return;
      window.Plotly.Plots.resize(el);
    } catch (error) {
      console.warn('Theme sync skipped for chart', el.id || el, error);
    }
  });
}


function forceChartThemeSync() {
  if (!window.Plotly) return;
  document.querySelectorAll('.js-plotly-plot, .plotly-graph-div, .dashboard-chart, [data-plotly-spec]').forEach((el) => {
    try {
      applyPlotlyThemeToElement(el);
      const box = el.getBoundingClientRect();
      if (!box.width || !box.height || el.offsetParent === null) return;
      window.Plotly.Plots.resize(el);
    } catch (error) {
      console.warn('forceChartThemeSync skipped', el.id || el, error);
    }
  });
}
function applyPlotlyThemeToAllCharts() {
  if (!window.Plotly) return;
  document.querySelectorAll('.js-plotly-plot, .plotly-graph-div, .dashboard-chart, [data-plotly-spec]').forEach((el) => {
    if (el && typeof applyPlotlyThemeToElement === 'function') {
      applyPlotlyThemeToElement(el);
    }
  });
}

function initThemeSwitcher() {
  let saved = 'midnight-blue';
  try { saved = localStorage.getItem('forecast-theme') || 'midnight-blue'; } catch (error) {}
  const normalized = normalizeThemeName(saved);
  const select = document.querySelector('[data-theme-select="true"]');
  if (select) {
    select.value = normalized;
    select.addEventListener('change', () => applyTheme(select.value));
  }
}

function initBackToTopButton() {
  const btn = document.getElementById('back-to-top');
  if (!btn) return;
  const sync = () => {
    const y = window.scrollY || document.documentElement.scrollTop || 0;
    btn.classList.toggle('is-visible', y > 420);
  };
  btn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
  window.addEventListener('scroll', sync, { passive: true });
  sync();
}

const AVAILABLE_VIEW_MODES = new Set(['analyst', 'human']);

function getCurrentViewMode() {
  const mode = document.documentElement.getAttribute('data-view-mode') || 'analyst';
  return AVAILABLE_VIEW_MODES.has(mode) ? mode : 'analyst';
}

function humanModeText(text) {
  const value = (text || '').trim();
  const swap = HUMAN_MODE_TEXT_SWAPS.find(([from]) => from === value);
  return swap ? swap[1] : text;
}

function updateHumanModeCopy(mode) {
  const human = mode === 'human';
  document.querySelectorAll('[data-original-text]').forEach((node) => {
    const original = node.getAttribute('data-original-text') || '';
    node.textContent = human ? humanModeText(original) : original;
  });
  const globalHelp = document.querySelector('.global-help');
  if (globalHelp) {
    globalHelp.innerHTML = human
      ? 'Human Mode is on. Use the simplified labels, the guide cards below, and the top help text for plain-English explanations.'
      : 'Hover the <span class="pill tiny">?</span> icons for quick explanations, then use the tips below to understand what matters most on each page.';
  }
}

function buildGlossaryRail(mode) {
  const rail = document.getElementById('page-tip-rail');
  if (!rail) return;
  rail.querySelectorAll('.mode-glossary-card').forEach((node) => node.remove());
  if (mode !== 'human') return;
  const tipGroup = rail.getAttribute('data-tip-group') || 'general';
  const glossaryKeys = tipGroup === 'results'
    ? ['roas', 'cohort', 'feature_importance', 'attribution', 'anomaly', 'confidence', 'whale', 'zscore']
    : ['forecast','roas','cohort'].filter(key => HUMAN_MODE_GLOSSARY.some(entry => entry.key === key));
  glossaryKeys.forEach((key) => {
    const entry = HUMAN_MODE_GLOSSARY.find((item) => item.key === key);
    if (!entry) return;
    const card = document.createElement('section');
    card.className = 'context-tip-card is-dynamic mode-glossary-card';
    card.innerHTML = `<div class="eyebrow">Human mode</div><strong>${entry.humanTitle}</strong><p>${entry.human}</p><div class="tip-kicker">🧠 Analyst term: ${entry.title}</div>`;
    rail.prepend(card);
  });
}

function refreshViewModePresentation() {
  const mode = getCurrentViewMode();
  updateHumanModeCopy(mode);
  buildGlossaryRail(mode);
}

function applyViewMode(mode) {
  const normalized = AVAILABLE_VIEW_MODES.has(mode) ? mode : 'analyst';
  const previous = getCurrentViewMode();
  document.documentElement.setAttribute('data-view-mode', normalized);
  document.body.classList.toggle('human-mode-active', normalized === 'human');
  const select = document.querySelector('[data-view-mode-select="true"]');
  if (select && select.value !== normalized) select.value = normalized;
  try { localStorage.setItem('forecast-view-mode', normalized); } catch (error) {}
  refreshViewModePresentation();

  const banner = document.getElementById('human-mode-banner');
  if (banner) {
    banner.style.display = normalized === 'human' ? 'flex' : 'none';
  }

  window.dispatchEvent(new CustomEvent('forecast-view-mode-change', {
    detail: { viewMode: normalized, previousViewMode: previous }
  }));
}

function initViewModeSwitcher() {
  let saved = 'analyst';
  try { saved = localStorage.getItem('forecast-view-mode') || 'analyst'; } catch (error) {}
  document.querySelectorAll('.stat-label, .results-tab-btn, .results-tab-copy h2, .card-header h2, .card-header h3, th').forEach((node) => {
    if (!node.getAttribute('data-original-text')) node.setAttribute('data-original-text', node.textContent.trim());
  });
  applyViewMode(saved || 'analyst');
  const select = document.querySelector('[data-view-mode-select="true"]');
  if (select) {
    if (AVAILABLE_VIEW_MODES.has(saved)) select.value = saved;
    select.addEventListener('change', () => applyViewMode(select.value));
  }
}


function parseServerTime(value){
  if(!value) return null;
  const hasZone = /Z$|[+-]\d\d:\d\d$/.test(value);
  return new Date(hasZone ? value : `${value}Z`);
}



function showForgeLoader(options = {}) {
  const overlay = document.getElementById('sf-loader');
  if (!overlay) return;
  const titleEl = document.getElementById('sf-loader-title');
  const textEl = document.getElementById('sf-loader-text');
  const phaseEl = document.getElementById('sf-loader-phase');
  const percentEl = document.getElementById('sf-loader-percent');
  const steps = Array.from(overlay.querySelectorAll('.sf-loader__steps span'));

  const title = options.title || 'Preparing your workspace';
  const text = options.text || 'Please wait while SignalForge gets everything ready.';
  const phase = options.phase || 'Processing';
  const activeStep = options.activeStep || 0;
  const progressPercent = Number.isFinite(Number(options.progressPercent)) ? Number(options.progressPercent) : ((activeStep + 1) / Math.max(steps.length, 1)) * 100;
  const isFinishLine = options.finishLine === true;

  if (titleEl) titleEl.textContent = title;
  if (textEl) textEl.textContent = text;
  if (phaseEl) phaseEl.textContent = phase;
  if (percentEl) percentEl.textContent = `${Math.round(progressPercent)}%`;

  steps.forEach((step, index) => {
    step.classList.toggle('is-active', index === activeStep);
    step.classList.toggle('is-complete', index < activeStep);
  });

  updateForgeLoaderProgress(progressPercent);

  overlay.classList.toggle('is-finish-line', isFinishLine);
  overlay.classList.add('is-visible');
  overlay.setAttribute('aria-hidden', 'false');

  if (window.__sfLoaderTicker) {
    window.clearInterval(window.__sfLoaderTicker);
    window.__sfLoaderTicker = null;
  }

  if (Array.isArray(options.messages) && options.messages.length > 1 && textEl) {
    let pointer = 0;
    window.__sfLoaderTicker = window.setInterval(() => {
      pointer = (pointer + 1) % options.messages.length;
      textEl.classList.add('is-swapping');
      window.setTimeout(() => {
        textEl.textContent = options.messages[pointer];
        textEl.classList.remove('is-swapping');
      }, 120);
    }, options.messageInterval || 1400);
  }
}

function updateForgeLoaderProgress(percent) {
  const overlay = document.getElementById('sf-loader');
  const percentEl = document.getElementById('sf-loader-percent');
  if (!overlay) return;
  const normalized = Math.max(0, Math.min(100, Number(percent) || 0));
  overlay.style.setProperty('--sf-loader-progress', String(normalized / 100));
  if (percentEl) percentEl.textContent = `${Math.round(normalized)}%`;
}

function hideForgeLoader() {
  const overlay = document.getElementById('sf-loader');
  if (!overlay) return;
  overlay.classList.remove('is-finish-line');
  overlay.classList.remove('is-visible');
  overlay.setAttribute('aria-hidden', 'true');
  if (window.__sfLoaderTicker) {
    window.clearInterval(window.__sfLoaderTicker);
    window.__sfLoaderTicker = null;
  }
}

function initForgeSubmitLoaders() {
  const forms = document.querySelectorAll('form[data-loader-submit]');
  forms.forEach((form) => {
    form.addEventListener('submit', (event) => {
      if (form.dataset.submitting === 'true') {
        event.preventDefault();
        return;
      }
      const type = form.getAttribute('data-loader-submit');
      const submitter = event.submitter || form.querySelector('[type="submit"]');
      if (submitter) {
        submitter.classList.add('is-loading');
        submitter.setAttribute('aria-disabled', 'true');
      }
      form.dataset.submitting = 'true';

      if (type === 'upload') {
        showForgeLoader({
          title: 'Uploading your data',
          text: 'SignalForge is collecting your files and validating the structure.',
          phase: 'Uploading files',
          activeStep: 0,
          progressPercent: 28,
          messages: [
            'SignalForge is collecting your files and validating the structure.',
            'Securing sheet names, file slots, and upload metadata.',
            'Preparing the next step so mapping opens cleanly.'
          ]
        });
      } else if (type === 'mapping') {
        showForgeLoader({
          title: 'Starting the mapping engine',
          text: 'We are locking in your column choices and preparing the aligned model table.',
          phase: 'Validating mapping',
          activeStep: 1,
          progressPercent: 62,
          messages: [
            'We are locking in your column choices and preparing the aligned model table.',
            'Checking dates, targets, helper fields, and smart bindings.',
            'Moving your project into the next stage of forecasting.'
          ]
        });
      } else {
        showForgeLoader();
      }
    });
  });
}


document.addEventListener('DOMContentLoaded', () => {
  initThemeSwitcher();
  initBackToTopButton();
  initDynamicTooltips();
  initViewModeSwitcher();
  initHoverCards();
  window.setTimeout(() => {
    initDynamicTooltips();
    initHoverCards();
  }, 120);
  initSidebarCenterpiece();
  initForgeSubmitLoaders();
  initUIFailsafes();
  initSidebarCurrentPage();
  window.setTimeout(() => {
    if (typeof applyPlotlyThemeToAllCharts === 'function') applyPlotlyThemeToAllCharts();
  }, 200);

  document.querySelectorAll('table').forEach(table => {
    if (!table.parentElement.classList.contains('table-scroll')) {
      const wrap = document.createElement('div');
      wrap.className = 'table-scroll';
      table.parentNode.insertBefore(wrap, table);
      wrap.appendChild(table);
    }
  });

  const mappingForm = document.querySelector('form[data-mapping-form="true"]');
  if (mappingForm) {
    const reloadControls = mappingForm.querySelectorAll('[data-mapping-reload="true"]');

    const applyReload = () => {
      const url = new URL(window.location.href);
      const formData = new FormData(mappingForm);

      // Persist ALL current form values so nothing is lost on reload.
      // Hidden fields and text inputs are written as query params.
      for (const [key, value] of formData.entries()) {
        if (value && typeof value === 'string') {
          url.searchParams.set(key, value);
        }
      }
      // Ensure run_id is always present
      const runId = formData.get('run_id');
      if (runId) url.searchParams.set('run_id', String(runId));

      window.location.href = url.toString();
    };

    reloadControls.forEach(control => control.addEventListener('change', applyReload));
  }

  const chartEls = Array.from(document.querySelectorAll('[data-plotly-spec]'));

  function ensureChartDebugPanel() {
    if (document.getElementById('chart-debug-panel')) return document.getElementById('chart-debug-panel');
    const gallery = document.querySelector('[data-results-section="charts"]');
    if (!gallery) return null;
    const panel = document.createElement('section');
    panel.id = 'chart-debug-panel';
    panel.className = 'card top-gap';
    panel.innerHTML = `
      <div class="card-header">
        <h2>Chart Debug Panel</h2>
        <div class="muted">Visual rendering diagnostics for this results page.</div>
      </div>
      <div class="metric-grid" style="margin-bottom:16px;">
        <div class="sim-box premium-sim-box"><span>Containers</span><strong id="chart-debug-total">0</strong></div>
        <div class="sim-box premium-sim-box"><span>Rendered</span><strong id="chart-debug-rendered">0</strong></div>
        <div class="sim-box premium-sim-box"><span>Failed</span><strong id="chart-debug-failed">0</strong></div>
        <div class="sim-box premium-sim-box"><span>Missing spec</span><strong id="chart-debug-missing">0</strong></div>
      </div>
      <div class="table-scroll">
        <table class="debug-table">
          <thead>
            <tr><th>Chart</th><th>Spec</th><th>Parse</th><th>Render</th><th>Error</th></tr>
          </thead>
          <tbody id="chart-debug-body"></tbody>
        </table>
      </div>
    `;
    gallery.insertAdjacentElement('afterend', panel);
    return panel;
  }

  const chartDebugState = [];
  function updateChartDebugPanel() {
    const panel = ensureChartDebugPanel();
    if (!panel) return;
    const body = panel.querySelector('#chart-debug-body');
    const total = chartDebugState.length;
    const rendered = chartDebugState.filter(r => r.rendered).length;
    const missing = chartDebugState.filter(r => !r.hasSpec).length;
    const failed = total - rendered;
    panel.querySelector('#chart-debug-total').textContent = String(total);
    panel.querySelector('#chart-debug-rendered').textContent = String(rendered);
    panel.querySelector('#chart-debug-failed').textContent = String(failed);
    panel.querySelector('#chart-debug-missing').textContent = String(missing);
    body.innerHTML = chartDebugState.map((row) => `
      <tr>
        <td>${row.title || row.id || 'Untitled'}</td>
        <td>${row.hasSpec ? 'Yes' : 'No'}</td>
        <td>${row.parsed ? 'Yes' : 'No'}</td>
        <td>${row.rendered ? 'Yes' : 'No'}</td>
        <td>${row.error || '—'}</td>
      </tr>
    `).join('');
  }

  
function rollingAverageSeries(values, windowSize = 30) {
  const out = [];
  const nums = (values || []).map(v => Number(v));
  for (let i = 0; i < nums.length; i++) {
    const start = Math.max(0, i - windowSize + 1);
    const slice = nums.slice(start, i + 1).filter(v => Number.isFinite(v));
    const avg = slice.length ? slice.reduce((a,b)=>a+b,0) / slice.length : null;
    out.push(avg);
  }
  return out;
}

function transformSpendingSlowdownSpec(spec, card) {
  if (!spec || !card) return spec;
  const titleNode = card.querySelector('h2');
  const subtitleNode = card.querySelector('.muted');
  const title = (titleNode?.textContent || '').trim().toLowerCase();
  const subtitle = (subtitleNode?.textContent || '').trim().toLowerCase();

  if (!title.includes('purchases per spender')) return spec;

  const next = JSON.parse(JSON.stringify(spec || {}));
  const trace = Array.isArray(next.data) && next.data.length ? next.data[0] : null;
  if (!trace || !Array.isArray(trace.y)) return next;

  const values = trace.y.map(v => Number(v));
  const allFlatOne = values.length && values.every(v => Number.isFinite(v) && Math.abs(v - 1) < 0.0001);
  const smoothed = rollingAverageSeries(values, 30);

  trace.y = smoothed;
  trace.name = 'Rolling Purchases / Spender';
  trace.mode = trace.mode || 'lines';
  trace.hovertemplate = 'Rolling purchases / spender: %{y:.2f}<extra></extra>';

  next.layout = next.layout || {};
  next.layout.title = {text: 'Rolling 30-Day Purchases per Spender'};
  next.layout.yaxis = {...(next.layout.yaxis || {}), title: {text: 'Rolling Purchases / Spender'}};

  if (titleNode) titleNode.textContent = 'Rolling 30-Day Purchases per Spender';
  if (subtitleNode) {
    subtitleNode.textContent = allFlatOne
      ? 'Original daily purchase-count logic was flat at 1.00, so SignalForge now shows a rolling-window view and flags that the uploaded revenue file may be aggregated.'
      : 'Rolling 30-day purchase count per active spender to better show repeat-buying behavior over time.';
  }
  card.dataset.chartTitle = 'rolling 30-day purchases per spender';
  if (allFlatOne) {
    const existing = card.querySelector('.p033-warning');
    if (!existing) {
      const warn = document.createElement('div');
      warn.className = 'insight-pill p033-warning';
      warn.textContent = 'P033 warning: this series was flat at 1.00 under the old daily logic, which often means the revenue file is aggregated or missing transaction-level purchase counts.';
      subtitleNode?.insertAdjacentElement('afterend', warn);
    }
  }
  return next;
}

chartEls.forEach((el) => {
    const card = el.closest('.chart-card');
    const entry = {
      id: el.id || '',
      title: (card && card.querySelector('h2') ? card.querySelector('h2').textContent.trim() : ''),
      hasSpec: false,
      parsed: false,
      rendered: false,
      error: ''
    };
    try {
      const rawSpec = String(el.dataset.plotlySpec || '').trim();
      entry.hasSpec = !!rawSpec;
      if (!rawSpec) {
        entry.error = 'Missing data-plotly-spec';
        chartDebugState.push(entry);
        return;
      }
      const sanitized = rawSpec
        .replace(/\bNaN\b/g, 'null')
        .replace(/\bInfinity\b/g, 'null')
        .replace(/-null/g, 'null')
        .replace(/\bundefined\b/g, 'null');
      let spec = JSON.parse(sanitized);
      const card = el.closest('.chart-card');
      spec = transformSpendingSlowdownSpec(spec, card);
      entry.parsed = true;
      el._plotlyOriginal = JSON.parse(JSON.stringify(spec));
      el._plotlyThemeSource = JSON.parse(JSON.stringify(spec));
      if (window.Plotly) {
        el.innerHTML = '';
        window.Plotly.newPlot(el, spec.data || [], spec.layout || {}, {responsive: true, displaylogo: false, modeBarButtonsToRemove: ['select2d','lasso2d']});
        el.dataset.chartRendered = '1';
        entry.rendered = true;
        if (typeof applyPlotlyThemeToElement === 'function') applyPlotlyThemeToElement(el);
      } else {
        entry.error = 'Plotly not available';
      }
    } catch (error) {
      entry.error = (error && error.message) ? error.message : String(error);
      console.warn('Skipping chart auto-render for element', el.id || el, error);
    } finally {
      chartDebugState.push(entry);
    }
  });
  updateChartDebugPanel();

  const chartSearch = document.getElementById('chart-search');
  if (chartSearch) {
    chartSearch.addEventListener('input', () => {
      const q = chartSearch.value.trim().toLowerCase();
      document.querySelectorAll('.chart-card').forEach(card => {
        const title = (card.querySelector('h2')?.textContent || '').toLowerCase();
        card.style.display = !q || title.includes(q) ? '' : 'none';
      });
    });
  }

  const chartHeight = document.getElementById('chart-height');
  const chartHeightMap = {standard: 420, tall: 560, compact: 320};
  const applyChartHeight = () => {
    const selected = chartHeight?.value || 'standard';
    const height = Number(chartHeightMap[selected] || selected || 420);
    chartEls.forEach(el => {
      el.style.minHeight = `${height}px`;
      if (window.Plotly && el.data) {
        window.Plotly.relayout(el, {height});
      }
    });
  };
  if (chartHeight) {
    chartHeight.addEventListener('change', applyChartHeight);
    applyChartHeight();
  }

  const chartWindow = document.getElementById('chart-date-window');
  if (chartWindow) {
    chartWindow.addEventListener('change', () => {
      const windowSize = chartWindow.value;
      chartEls.forEach(el => {
        const original = el._plotlyOriginal;
        if (!original || !window.Plotly) return;
        if (windowSize === 'all') {
          window.Plotly.react(el, original.data || [], original.layout || {}, {responsive: true, displaylogo: false});
          if (typeof applyPlotlyThemeToElement === 'function') applyPlotlyThemeToElement(el);
          return;
        }
        const n = Number(windowSize);
        const data = (original.data || []).map(trace => {
          const cloned = {...trace};
          if (Array.isArray(trace.x)) cloned.x = trace.x.slice(-n);
          if (Array.isArray(trace.y)) cloned.y = trace.y.slice(-n);
          return cloned;
        });
        window.Plotly.react(el, data, original.layout || {}, {responsive: true, displaylogo: false});
        if (typeof applyPlotlyThemeToElement === 'function') applyPlotlyThemeToElement(el);
      });
      applyChartHeight();
    });
  }

  
  const refreshBtn = document.getElementById('refresh-layout-btn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
      applyChartHeight();
      chartEls.forEach(el => {
        if (!window.Plotly || !el || !el.data) return;
        try {
          const box = el.getBoundingClientRect();
          if (!box.width || !box.height || el.offsetParent === null) return;
          window.Plotly.Plots.resize(el);
          if (typeof applyPlotlyThemeToElement === 'function') applyPlotlyThemeToElement(el);
        } catch (error) {
          console.warn('Refresh layout skipped for chart', el.id || el, error);
        }
      });
      window.dispatchEvent(new Event('resize'));
    });
  }


  document.querySelectorAll('[data-filter-table]').forEach(table => {
    const inputId = table.getAttribute('data-filter-table');
    const input = document.getElementById(inputId);
    if (!input) return;
    input.addEventListener('input', () => {
      const q = input.value.trim().toLowerCase();
      table.querySelectorAll('tbody tr').forEach(tr => {
        const text = tr.textContent.toLowerCase();
        tr.style.display = !q || text.includes(q) ? '' : 'none';
      });
    });
  });

  const shell = document.querySelector('[data-progress-url]');
  if (!shell) return;
  const progressUrl = shell.dataset.progressUrl;
  const resultUrl = shell.dataset.resultUrl;
  const stepEl = document.getElementById('progress-step');
  const detailEl = document.getElementById('progress-detail');
  const percentEl = document.getElementById('progress-percent');
  const barEl = document.getElementById('progress-bar');
  const statusEl = document.getElementById('progress-status');
  const updatedEl = document.getElementById('progress-updated');
  const resultEl = document.getElementById('progress-result');
  const messageEl = document.getElementById('progress-message');
  const render = (state) => {
    const percent = Number(state.percent || 0);
    stepEl.textContent = state.step || 'Working';
    detailEl.textContent = state.detail || '';
    percentEl.textContent = `${percent}%`;
    barEl.style.width = `${percent}%`;
    statusEl.textContent = (state.status || 'queued').replace(/_/g, ' ');
    updatedEl.textContent = state.updated_at ? parseServerTime(state.updated_at).toLocaleTimeString([], {hour:'numeric', minute:'2-digit', second:'2-digit'}) : '—';
    if (state.status === 'completed') {
      resultEl.textContent = 'Ready';
      messageEl.innerHTML = '<div class="status-ok">Forecast complete. Opening results…</div>';
      window.setTimeout(() => { window.location.href = state.result_url || resultUrl; }, 700);
      return true;
    }
    if (state.status === 'failed') {
      resultEl.textContent = 'Failed';
      messageEl.innerHTML = `<div class="status-error">${state.error || 'The run failed.'}</div>`;
      return true;
    }
    resultEl.textContent = 'Processing';
    messageEl.innerHTML = '<div class="status-warn">The page updates automatically while your models run.</div>';
    return false;
  };
  const poll = async () => {
    try {
      const response = await fetch(progressUrl, { cache: 'no-store' });
      const state = await response.json();
      const done = render(state);
      if (!done) window.setTimeout(poll, 1200);
    } catch (error) {
      messageEl.innerHTML = '<div class="status-error">Could not refresh progress right now. Retrying…</div>';
      window.setTimeout(poll, 2000);
    }
  };
  poll();
});


const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
if (!document.documentElement.getAttribute('data-theme')) {
  try {
    const existingTheme = localStorage.getItem('forecast-theme');
    if (!existingTheme || !AVAILABLE_THEMES.has(normalizeThemeName(existingTheme))) {
      applyTheme(prefersDark ? 'midnight-blue' : 'arctic-glass', { silent: true, force: true });
    } else {
      applyTheme(existingTheme, { silent: true, force: true });
    }
  } catch (error) {
    applyTheme(prefersDark ? 'midnight-blue' : 'arctic-glass', { silent: true, force: true });
  }
}
