const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;
const formatMs = (value) => `${value.toFixed(2)} ms`;
const formatBytes = (value) => `${(value / 1024 / 1024).toFixed(2)} MB`;

function renderMetrics(metrics) {
  document.querySelector('#metric-finetuned').textContent = formatPercent(metrics.finetuned_f1);
  document.querySelector('#metric-baseline').textContent = formatPercent(metrics.baseline_f1);
  document.querySelector('#metric-size').textContent = formatBytes(metrics.onnx_int8_bytes);
  document.querySelector('#metric-latency').textContent = formatMs(metrics.onnx_int8_latency_ms_median);
  const rows = [
    ['Simple method (model not trained on bottles)', metrics.baseline_f1, null, null],
    ['Bottle-trained model, full size', metrics.finetuned_f1, metrics.pytorch_model_bytes, metrics.pytorch_latency_ms_median],
    ['Bottle-trained model, compressed', metrics.int8_f1, metrics.onnx_int8_bytes, metrics.onnx_int8_latency_ms_median],
  ];
  document.querySelector('#metrics-body').innerHTML = rows.map(([name, f1, size, latency]) => `<tr><td>${name}</td><td>${formatPercent(f1)}</td><td>${size ? formatBytes(size) : 'not measured'}</td><td>${latency ? formatMs(latency) : 'not measured'}</td></tr>`).join('');
}

function renderDemo(results) {
  const select = document.querySelector('#demo-select');
  const items = results.demo;
  select.innerHTML = items.map((item) => `<option value="${item.id}">Photo ${item.id + 1}: ${item.actual}</option>`).join('');
  document.querySelector('#demo-count').textContent = `${items.length} sample photos`;
  document.querySelector('#miss-count').textContent = `Wrong answers on test photos: ${results.misclassified.length}`;
  document.querySelector('#errors-grid').innerHTML = results.misclassified.length ? results.misclassified.map((item) => `<article class="result-card"><img src="${item.image}" alt="Test photo the model got wrong" style="width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:12px"><span class="eyebrow">Really ${item.actual}, model said ${item.predicted}</span><strong>Model was ${formatPercent(item.confidence)} sure</strong></article>`).join('') : '<p class="prose">No wrong answers on the test photos in this run.</p>';
  const update = () => {
    const item = items.find((candidate) => candidate.id === Number(select.value));
    const correct = item.actual === item.predicted;
    document.querySelector('#demo-image').src = item.image;
    document.querySelector('#demo-prediction').textContent = item.predicted === 'defective' ? 'Defective' : 'Acceptable';
    document.querySelector('#demo-confidence').textContent = `Model was ${formatPercent(item.confidence)} sure`;
    document.querySelector('#demo-explanation').textContent = correct ? `Right answer. This bottle really is ${item.actual}.` : `Wrong answer. This bottle really is ${item.actual}.`;
  };
  select.addEventListener('change', update);
  update();
}

fetch('data/results.json').then((response) => response.json()).then((results) => {
  renderMetrics(results.metrics);
  renderDemo(results);
}).catch(() => {
  document.querySelector('#demo-explanation').textContent = 'The saved results could not be loaded.';
});
