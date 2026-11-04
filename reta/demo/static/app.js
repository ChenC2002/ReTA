const $ = id => document.getElementById(id);
let metadata, historyResult, activeVisit = 0;
const percent = value => `${(value * 100).toFixed(1)}%`;

function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}
async function request(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Check the visit codes and try again.');
  return payload;
}

function setVisits(visits) {
  $('visits').replaceChildren();
  visits.forEach(codes => addVisit(codes));
  $('history-title').textContent = 'Visit history';
}

function markDirty() {
  historyResult = null;
  $('output').hidden = true;
  $('empty-state').hidden = false;
  $('visit-tabs').replaceChildren();
  $('latency').textContent = 'Ready when you are';
  $('error').hidden = true;
  $('history-title').textContent = 'Edited visit history';
}

function renumber() {
  [...$('visits').children].forEach((el, i) => {
    el.querySelector('label span').textContent = `VISIT ${String(i + 1).padStart(2, '0')}`;
    el.querySelector('input').setAttribute('aria-label', `Visit ${i + 1} ICD-9-CM codes`);
    el.querySelector('button').setAttribute('aria-label', `Remove visit ${i + 1}`);
    el.querySelector('button').disabled = $('visits').children.length < 2;
  });
  $('add-visit').disabled = $('visits').children.length >= 6;
}

function addVisit(codes = []) {
  const wrapper = node('div', undefined, 'visit-input');
  const label = node('label'),
    title = node('span'),
    remove = node('button', 'Remove', 'remove');
  remove.type = 'button';
  remove.onclick = () => {
    wrapper.remove();
    renumber();
    markDirty();
  };
  label.append(title, remove);
  const input = node('input');
  input.value = codes.join(', ');
  input.placeholder = '25000, 4019';
  input.autocomplete = 'off';
  input.oninput = markDirty;
  wrapper.append(label, input);
  $('visits').append(wrapper);
  renumber();
  markDirty();
}

function drawGraph(graph) {
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 320 180');
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', `${graph.nodes.length} graph nodes and ${graph.edges.length} connections`);
  const positions = new Map();
  for (const [kind, radius] of [
      ['ontology', 43],
      ['observed', 73],
      ['imported', 73]
    ]) {
    const group = graph.nodes.filter(n => n.kind === kind);
    group.forEach((n, i) => {
      const angle = (i / Math.max(1, group.length)) * 2 * Math.PI - Math.PI / 2 + (kind === 'imported' ? .2 : 0);
      positions.set(n.id, [160 + Math.cos(angle) * radius * 1.5, 90 + Math.sin(angle) * radius]);
    });
  }
  for (const [a, b] of graph.edges) {
    const pa = positions.get(a),
      pb = positions.get(b);
    if (!pa || !pb) continue;
    const line = document.createElementNS(NS, 'line');
    line.setAttribute('x1', pa[0]);
    line.setAttribute('y1', pa[1]);
    line.setAttribute('x2', pb[0]);
    line.setAttribute('y2', pb[1]);
    line.setAttribute('stroke', '#d9e2d7');
    line.setAttribute('stroke-width', '.9');
    svg.append(line);
  }
  for (const n of graph.nodes) {
    const [x, y] = positions.get(n.id);
    const circle = document.createElementNS(NS, 'circle');
    circle.setAttribute('cx', x);
    circle.setAttribute('cy', y);
    circle.setAttribute('r', n.kind === 'observed' ? 5 : 3.8);
    circle.setAttribute('fill', {
      observed: '#24434d',
      ontology: '#b6c5b9',
      imported: '#1b9b75'
    } [n.kind]);
    const title = document.createElementNS(NS, 'title');
    title.textContent = `${n.code}: ${n.name}`;
    circle.append(title);
    circle.setAttribute('tabindex', '0');
    circle.setAttribute('role', 'button');
    circle.setAttribute('aria-label', `${n.code}: ${n.name}`);
    const inspect = () => { $('graph-selection').textContent = `${n.name} · ${n.code}`; };
    circle.onclick = inspect;
    circle.onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); inspect(); } };
    svg.append(circle);
  }
  $('graph').replaceChildren(svg);
  $('graph-counts').textContent = `${graph.nodes.length} nodes · ${graph.edges.length} connections · select a node to inspect`;
}

function renderVisit(index) {
  activeVisit = index;
  const step = historyResult.steps[index];
  $('prediction-context').textContent = `AFTER VISIT ${step.visit}`;
  $('observed-summary').replaceChildren(node('b', 'Observed: '), document.createTextNode(step.codes.map(c => c.name).join(' · ')));
  $('graph-selection').textContent = 'Select a node to inspect its concept.';
  $('retrieval-description').textContent = metadata.grounded_candidates_only
    ? 'Roots matched to observed visit concepts, then ranked by code similarity.'
    : 'Ranked by code and history similarity.';
  [...$('visit-tabs').children].forEach((tab, i) => {
    tab.classList.toggle('active', i === index);
    tab.setAttribute('aria-selected', i === index);
    tab.tabIndex = i === index ? 0 : -1;
  });
  $('retrieval-count').textContent = `${step.candidates.length} candidates`;
  $('candidates').replaceChildren();
  let shown = step.candidates.slice(0, 5);
  const selected = step.candidates.find(c => c.selected);
  if (selected && !shown.includes(selected)) shown = [...shown.slice(0, 4), selected];
  if (!shown.length) $('candidates').append(node('p', metadata.has_policy ? 'No grounded candidate for this visit. The policy can only Skip.' : 'This checkpoint uses the base encoder without a policy.', 'small muted'));
  for (const candidate of shown) {
    const rank = step.candidates.indexOf(candidate) + 1;
    const row = node('div', undefined, `candidate${candidate.selected ? ' selected' : ''}`);
    const body = node('div');
    body.append(node('div', candidate.name, 'candidate-name'), node('div', `${candidate.nodes} knowledge nodes${candidate.selected ? ' · selected' : ''}`, 'candidate-meta'));
    const definition = node('details', undefined, 'candidate-definition');
    definition.append(node('summary', 'View template'), node('p', candidate.definition));
    body.append(definition);
    row.append(node('span', String(rank).padStart(2, '0'), 'rank'), body, node('span', candidate.score.toFixed(3), 'score'));
    $('candidates').append(row);
  }
  const actions = {
    soft: ['Soft import', 'A retrieved template adjusts the observed code features. Graph structure stays the same.'],
    hard: ['Hard import', `${step.added_nodes} knowledge nodes are connected to the current visit graph.`],
    hard_noop: ['Hard import · no change', 'The selected template adds no connected structure to this visit.'],
    skip: ['Skip', 'The policy keeps this visit unchanged.']
  };
  const [title, description] = actions[step.action];
  $('action').textContent = title;
  $('action').className = `action ${step.action.startsWith('hard') ? 'hard' : step.action}`;
  $('action-description').textContent = description;
  $('decision-summary').textContent = `${title}${selected ? ' · ' + selected.name : ''}`;
  $('decision-detail').textContent = description;
  const compare = $('compare').checked;
  $('score-heading').textContent = compare ? 'BEFORE → AFTER IMPORT' : 'MODEL SCORE';
  $('comparison-note').hidden = !compare;
  drawGraph(step.graph);
  $('predictions').replaceChildren();
  for (const item of step.predictions) {
    const row = node('div', undefined, 'prediction-row'),
      name = node('div', item.name);
    name.append(node('span', `CCS ${item.code}`, 'diagnosis-code'));
    const track = node('div', undefined, 'bar-track'),
      after = node('div', undefined, 'bar-after'),
      before = node('div', undefined, 'bar-before');
    after.style.width = percent(item.probability);
    before.style.left = percent(item.before);
    before.hidden = !compare;
    track.append(after, before);
    const values = node('div', compare ? `${percent(item.before)} → ` : '', 'prediction-values');
    values.append(node('strong', percent(item.probability)));
    row.append(name, track, values);
    $('predictions').append(row);
  }
}
async function predict() {
  $('error').hidden = true;
  $('run').disabled = true;
  $('run').textContent = 'Running model…';
  $('latency').textContent = 'Running model…';
  $('example').disabled = true;
  $('add-visit').disabled = true;
  $('visits').querySelectorAll('input,button').forEach(el => {
    el.disabled = true;
  });
  try {
    const visits = [...$('visits').querySelectorAll('input')].map(input => input.value.split(/[,;\s]+/).filter(Boolean));
    historyResult = await request('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        visits,
        top_k: 8
      })
    });
    $('empty-state').hidden = true;
    $('output').hidden = false;
    $('visit-tabs').replaceChildren();
    historyResult.steps.forEach((step, i) => {
      const button = node('button', `Visit ${i + 1}`, 'visit-tab');
      button.setAttribute('role', 'tab');
      button.onclick = () => renderVisit(i);
      button.onkeydown = event => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        const count = historyResult.steps.length;
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? count - 1
          : (i + (event.key === 'ArrowRight' ? 1 : -1) + count) % count;
        renderVisit(next);
        $('visit-tabs').children[next].focus();
      };
      $('visit-tabs').append(button);
    });
    $('latency').textContent = `${historyResult.elapsed_ms} ms · ${historyResult.steps.length} visits`;
    renderVisit(historyResult.steps.length - 1);
    $('history-title').textContent = `${historyResult.steps.length} completed visits · edit history`;
    if (window.matchMedia('(max-width: 760px)').matches) $('history-editor').open = false;
  } catch (error) {
    $('latency').textContent = 'Prediction unavailable';
    $('history-editor').open = true;
    $('error').textContent = error.message;
    $('error').hidden = false;
  } finally {
    $('run').disabled = false;
    $('example').disabled = false;
    $('visits').querySelectorAll('input').forEach(el => {
      el.disabled = false;
    });
    renumber();
    $('run').replaceChildren(document.createTextNode('Run prediction '), node('span', '↗'));
  }
}

function renderExperiment(results) {
  if (!results || !results.metrics) return;
  $('experiment').hidden = false;
  $('experiment-context').textContent = results.context || 'Patient-disjoint evaluation · next-visit CCS diagnosis prediction';
  for (const [key, value] of Object.entries(results.metrics)) {
    const row = node('tr', undefined, key === 'stage2' ? 'highlight' : '');
    row.append(node('td', {
      training_prevalence: 'Diagnosis frequency',
      stage1: 'Stage 1',
      stage2: 'ReTA · Stage 2'
    } [key] || key));
    for (const metric of ['AUPRC_macro', 'AUPRC_micro', 'MicroF1', 'Acc@20']) row.append(node('td', `${(value[metric] * 100).toFixed(2)}%`));
    $('metrics').append(row);
  }
}
$('add-visit').onclick = () => addVisit();
$('run').onclick = predict;
$('example').onchange = () => {
  setVisits(metadata.examples[Number($('example').value)].visits);
  predict();
};
$('compare').onchange = () => { if (historyResult) renderVisit(activeVisit); };
let searchTimer, searchSequence = 0;
$('search').oninput = () => {
  clearTimeout(searchTimer);
  const sequence = ++searchSequence;
  searchTimer = setTimeout(async () => {
    try {
      const items = await request(`/api/concepts?q=${encodeURIComponent($('search').value)}`);
      if (sequence !== searchSequence) return;
      $('search-results').replaceChildren();
      for (const item of items) {
        const row = node('button', undefined, 'search-item');
        row.type = 'button';
        row.title = 'Add to the last visit';
        row.onclick = () => {
          const input = [...$('visits').querySelectorAll('input')].at(-1);
          if (!input || input.disabled) return;
          const codes = input.value.split(/[,;\s]+/).filter(Boolean);
          if (!codes.includes(item.code)) codes.push(item.code);
          input.value = codes.join(', ');
          markDirty();
        };
        row.append(node('b', item.code), document.createTextNode(item.name));
        $('search-results').append(row);
      }
    } catch {
      $('search-results').textContent = 'Search unavailable.';
    }
  }, 250);
};
(async () => {
  try {
    metadata = await request('/api/meta');
    metadata.examples.forEach((example, index) => {
      const option = node('option', example.name);
      option.value = index;
      $('example').append(option);
    });
    setVisits(metadata.examples[0].visits);
    $('example').disabled = false;
    $('run').disabled = false;
    $('connection').textContent = 'Model connected';
    $('connection').className = 'status live';
    renderExperiment(metadata.results);
    await predict();
  } catch {
    $('connection').textContent = 'Model unavailable';
    $('error').textContent = 'The model service is unavailable. Check the server, then reload this page.';
    $('error').hidden = false;
  }
})();
