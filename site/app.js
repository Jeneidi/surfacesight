const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;
const formatMs = (value) => `${value.toFixed(2)} ms`;
const formatBytes = (value) => `${(value / 1024 / 1024).toFixed(2)} MB`;

function renderMetrics(metrics) {
  document.querySelector('#metric-finetuned').textContent = formatPercent(metrics.finetuned_f1);
  document.querySelector('#metric-baseline').textContent = formatPercent(metrics.baseline_f1);
  document.querySelector('#metric-size').textContent = formatBytes(metrics.onnx_int8_bytes);
  document.querySelector('#metric-latency').textContent = formatMs(metrics.onnx_int8_latency_ms_median);
  const rows = [
    ['Frozen ImageNet feature baseline', metrics.baseline_f1, null, null],
    ['Fine-tuned ResNet18', metrics.finetuned_f1, metrics.pytorch_model_bytes, metrics.pytorch_latency_ms_median],
    ['Fine-tuned ONNX int8', metrics.int8_f1, metrics.onnx_int8_bytes, metrics.onnx_int8_latency_ms_median],
  ];
  document.querySelector('#metrics-body').innerHTML = rows.map(([name, f1, size, latency]) => `<tr><td>${name}</td><td>${formatPercent(f1)}</td><td>${size ? formatBytes(size) : 'feature baseline'}</td><td>${latency ? formatMs(latency) : 'not exported'}</td></tr>`).join('');
}

function renderDemo(results) {
  const select = document.querySelector('#demo-select');
  const items = results.demo;
  select.innerHTML = items.map((item) => `<option value="${item.id}">Example ${item.id + 1}: ${item.actual}</option>`).join('');
  document.querySelector('#demo-count').textContent = `${items.length} recorded examples`;
  document.querySelector('#miss-count').textContent = `${results.misclassified.length} held-out mistakes to inspect`;
  document.querySelector('#errors-grid').innerHTML = results.misclassified.length ? results.misclassified.map((item) => `<article class="result-card"><img src="${item.image}" alt="Misclassified held-out bottle image" style="width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:12px"><span class="eyebrow">${item.actual} predicted as ${item.predicted}</span><strong>${formatPercent(item.confidence)} confidence</strong></article>`).join('') : '<p class="prose">No held-out errors in this run.</p>';
  const update = () => {
    const item = items.find((candidate) => candidate.id === Number(select.value));
    const correct = item.actual === item.predicted;
    document.querySelector('#demo-image').src = item.image;
    document.querySelector('#demo-prediction').textContent = item.predicted === 'defective' ? 'Defective' : 'Acceptable';
    document.querySelector('#demo-confidence').textContent = `${formatPercent(item.confidence)} confidence`;
    document.querySelector('#demo-explanation').textContent = correct ? `Correct call. Actual label: ${item.actual}.` : `Missed call. Actual label: ${item.actual}.`;
  };
  select.addEventListener('change', update);
  update();
}

fetch('data/results.json').then((response) => response.json()).then((results) => {
  renderMetrics(results.metrics);
  renderDemo(results);
}).catch(() => {
  document.querySelector('#demo-explanation').textContent = 'Recorded results could not be loaded.';
});
