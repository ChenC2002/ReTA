const $ = id => document.getElementById(id);
let metadata, historyResult;
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
}

function markDirty() {
  historyResult = null;
  $('output').hidden = true;
  $('empty-state').hidden = false;
  $('visit-tabs').replaceChildren();
  $('latency').textContent = 'Ready when you are';
  $('error').hidden = true;
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
  svg.setAttribute('role', 'img');
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
    svg.append(circle);
  }
  $('graph').replaceChildren(svg);
  $('graph-counts').textContent = `${graph.nodes.length} nodes · ${graph.edges.length} connections · hover to inspect`;
}

function renderVisit(index) {
  const step = historyResult.steps[index];
  [...$('visit-tabs').children].forEach((tab, i) => {
    tab.classList.toggle('active', i === index);
    tab.setAttribute('aria-selected', i === index);
  });
  $('retrieval-count').textContent = `${step.candidates.length} candidates`;
  $('candidates').replaceChildren();
  let shown = step.candidates.slice(0, 5);
  const selected = step.candidates.find(c => c.selected);
  if (selected && !shown.includes(selected)) shown = [...shown.slice(0, 4), selected];
  if (!shown.length) $('candidates').append(node('p', 'This checkpoint uses the base encoder without a policy.', 'small muted'));
  for (const candidate of shown) {
    const rank = step.candidates.indexOf(candidate) + 1;
    const row = node('div', undefined, `candidate${candidate.selected ? ' selected' : ''}`);
    const body = node('div');
    body.append(node('div', candidate.name, 'candidate-name'), node('div', `${candidate.nodes} knowledge nodes${candidate.selected ? ' · selected' : ''}`, 'candidate-meta'));
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
    track.append(after, before);
    const values = node('div', `${percent(item.before)} → `, 'prediction-values');
    values.append(node('strong', percent(item.probability)));
    row.append(name, track, values);
    $('predictions').append(row);
  }
}
async function predict() {
  $('error').hidden = true;
  $('run').disabled = true;
  $('run').textContent = 'Running model…';
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
      $('visit-tabs').append(button);
    });
    $('latency').textContent = `${historyResult.elapsed_ms} ms · ${historyResult.steps.length} visits`;
    renderVisit(historyResult.steps.length - 1);
  } catch (error) {
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
$('example').onchange = () => setVisits(metadata.examples[Number($('example').value)].visits);
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
        const row = node('div', undefined, 'search-item');
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
  } catch {
    $('connection').textContent = 'Model unavailable';
    $('error').textContent = 'The model service is unavailable. Check the server, then reload this page.';
    $('error').hidden = false;
  }
})();
