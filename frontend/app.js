const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const demoResult = {
  finding: { title: 'Right lower lobe opacity', summary: 'Pattern is most consistent with a focal air-space opacity; correlate with symptoms and prior imaging.' },
  trust: { status: 'trusted', label: 'Trusted', confidence: 0.86, uncertainty: 0.18, note: 'Above review threshold' },
  reliability: {
    image: { score: 0.88, label: 'High reliability', copy: 'Adequate exposure, positioning, and field coverage for model interpretation.', quality: 'Passed 6 / 6 checks' },
    clinical: { score: 0.62, label: 'Moderate reliability', copy: 'Symptoms and vitals support interpretation; no comparison study supplied.', quality: 'Context 4 / 6 fields' }
  },
  fusion: { image: 0.67, clinical: 0.33, note: 'The image carries more weight because acquisition quality is strong and the clinical context is partially complete.' },
  heatmap: 'Attention is concentrated in the right lower lung zone, with secondary signal at the left base.',
  explanation: 'The focal lower-zone opacity aligns with the reported productive cough and fever. Heart rate and oxygen saturation increase the urgency of clinician review, while AP positioning adds modest uncertainty.',
  evidence: [
    { title: 'Approach to the adult with a focal pulmonary opacity', source: 'American College of Radiology · 2024', citation: '[1]' },
    { title: 'Radiographic patterns of community-acquired pneumonia', source: 'Chest Imaging Review · 2023', citation: '[2]' },
    { title: 'Calibrating AI assistance in chest radiography', source: 'Radiology: AI · 2024', citation: '[3]' }
  ],
  audit: [
    { time: 'NOW', text: 'Multimodal result generated · model v0.9' },
    { time: '—', text: 'Clinical context vector assembled from 7 inputs' },
    { time: '—', text: 'Image quality checks completed · 6 / 6 passed' }
  ],
  guidance: {
    headline: 'Demonstration mode — clinician review required',
    plain_summary: 'The application needs a validated local checkpoint before its image signal can be considered meaningful.',
    urgency: { level: 'prompt_review', title: 'Clinician review required', message: 'Use this screen only as a workflow demonstration.', reasons: [] },
    checklist: ['Assess the patient and vital signs first.', 'Review the original radiograph and compare with prior imaging.', 'Use a qualified clinician or radiologist to make the final interpretation.'],
    possible_tests: [{ title: 'No automatic test recommendation', text: 'Choose additional investigations only after clinical assessment and a specific diagnostic question.' }],
    treatment: { title: 'Medication and prescription safety', status: 'No prescription generated', message: 'Do not start, stop, or dose medication from this AI output.', clinician_action: 'Document the diagnosis and treatment rationale independently.' },
    model_language: { positive_signal: 'A positive signal deserves review; it does not confirm a condition.', negative_signal: 'A negative signal does not rule out disease.', abstain: 'Insufficient support means the model is withholding a call.' },
    references: []
  }
};

const els = {
  dropzone: $('#dropzone'), input: $('#image-input'), fileStatus: $('#file-status'), form: $('#clinical-form'), analyze: $('#analyze-button'),
  empty: $('#empty-state'), content: $('#result-content'), toast: $('#toast')
};
let selectedFile = null;

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add('show');
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => els.toast.classList.remove('show'), 3600);
}

function setFile(file) {
  if (!file) return;
  const maxBytes = 25 * 1024 * 1024;
  if (file.size > maxBytes) {
    els.fileStatus.className = 'file-status error';
    els.fileStatus.innerHTML = '<span class="status-dot"></span><span>File exceeds the 25 MB limit</span>';
    showToast('Please choose an image smaller than 25 MB.');
    return;
  }
  selectedFile = file;
  els.fileStatus.className = 'file-status ready';
  els.fileStatus.innerHTML = `<span class="status-dot"></span><span>${escapeHtml(file.name)} · ${formatBytes(file.size)}</span>`;
  const stage = $('.xray-stage');
  if (file instanceof File && file.type.startsWith('image/')) {
    if (stage.dataset.objectUrl) URL.revokeObjectURL(stage.dataset.objectUrl);
    stage.dataset.objectUrl = URL.createObjectURL(file);
    stage.style.backgroundImage = `linear-gradient(rgba(212, 229, 232, .3), rgba(212, 229, 232, .3)), url(${stage.dataset.objectUrl})`;
    stage.style.backgroundSize = 'cover';
    stage.style.backgroundPosition = 'center';
  }
}

function formatBytes(bytes) { return bytes < 1024 * 1024 ? `${Math.round(bytes / 1024)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`; }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char])); }

els.dropzone.addEventListener('click', () => els.input.click());
els.dropzone.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); els.input.click(); } });
els.input.addEventListener('change', (event) => setFile(event.target.files[0]));
['dragenter', 'dragover'].forEach((type) => els.dropzone.addEventListener(type, (event) => { event.preventDefault(); els.dropzone.classList.add('dragover'); }));
['dragleave', 'drop'].forEach((type) => els.dropzone.addEventListener(type, (event) => { event.preventDefault(); els.dropzone.classList.remove('dragover'); }));
els.dropzone.addEventListener('drop', (event) => setFile(event.dataTransfer.files[0]));

function formDataObject() {
  const data = Object.fromEntries(new FormData(els.form).entries());
  if (data.age) data.age = Number(data.age);
  ['heart_rate', 'oxygen_saturation', 'temperature'].forEach((key) => { if (data[key]) data[key] = Number(data[key]); });
  $$('#clinical-form input[type="checkbox"]').forEach((input) => { data[input.name] = input.checked; });
  return data;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || payload.message || `Request failed (${response.status})`);
  return payload;
}

function normalizeResult(payload) {
  const result = payload?.result || payload?.data || payload;
  const get = (object, keys, fallback) => keys.reduce((value, key) => value ?? object?.[key], undefined) ?? fallback;
  const labels = Array.isArray(result.labels) ? result.labels : [];
  const topLabel = labels.slice().sort((a, b) => Number(b.probability || 0) - Number(a.probability || 0))[0];
  const quality = result.quality || {};
  const clinicalBlock = result.clinical || {};
  const actualReliability = {
    image: { score: Number(quality.score || 0), label: quality.score >= .7 ? 'High reliability' : 'Review quality', copy: `Contrast ${(Number(quality.contrast || 0) * 100).toFixed(0)}% · sharpness ${(Number(quality.sharpness || 0) * 100).toFixed(0)}% · size check ${(Number(quality.size || 0) * 100).toFixed(0)}%`, quality: `${quality.width || 0} × ${quality.height || 0}px` },
    clinical: { score: Number(clinicalBlock.completeness_score || 0), label: clinicalBlock.completeness_score >= .7 ? 'High reliability' : 'Incomplete context', copy: 'Completeness of the submitted clinical fields.', quality: `${(clinicalBlock.fields_received || []).length} fields received` }
  };
  const actualTrust = { status: String(result.overall_decision || '').includes('abstain') ? 'review' : 'trusted', label: String(result.overall_decision || '').includes('abstain') ? 'Review needed' : 'Trusted', confidence: topLabel ? Number(topLabel.probability || 0) : 0, uncertainty: Number(result.predictive_uncertainty ?? (topLabel ? 1 - Number(topLabel.probability || 0) : 1)), note: result.disclaimer || 'Review all signals with the treating clinician.' };
  const guidance = result.clinical_guidance || demoResult.guidance;
  const finding = result.finding || result.prediction || {};
  const trust = result.trust || result.confidence || actualTrust;
  const reliability = result.reliability || actualReliability;
  const fusion = result.fusion || result.weights || { image_weight: result.fusion?.image_weight_wi, clinical_weight: result.fusion?.clinical_weight_wc, note: result.fusion?.formula };
  return {
    finding: { title: get(finding, ['title', 'label', 'name'], guidance.headline || topLabel?.label || get(result, ['predicted_finding', 'prediction_label'], demoResult.finding.title)), summary: get(finding, ['summary', 'description'], guidance.plain_summary || (topLabel ? `${topLabel.label} signal ${(Number(topLabel.probability || 0) * 100).toFixed(1)}%; clinician review required.` : demoResult.finding.summary)) },
    trust: { status: get(trust, ['status'], get(result, ['status'], demoResult.trust.status)), label: get(trust, ['label'], demoResult.trust.label), confidence: Number(get(trust, ['confidence', 'calibrated_confidence'], demoResult.trust.confidence)), uncertainty: Number(get(trust, ['uncertainty', 'predictive_uncertainty'], demoResult.trust.uncertainty)), note: get(trust, ['note'], demoResult.trust.note) },
    reliability: { image: { score: Number(get(reliability.image || result.image_reliability, ['score', 'value'], demoResult.reliability.image.score)), label: get(reliability.image || result.image_reliability, ['label', 'status'], demoResult.reliability.image.label), copy: get(reliability.image || result.image_reliability, ['copy', 'explanation'], demoResult.reliability.image.copy), quality: get(reliability.image || result.image_reliability, ['quality', 'quality_note'], demoResult.reliability.image.quality) }, clinical: { score: Number(get(reliability.clinical || result.clinical_reliability, ['score', 'value'], demoResult.reliability.clinical.score)), label: get(reliability.clinical || result.clinical_reliability, ['label', 'status'], demoResult.reliability.clinical.label), copy: get(reliability.clinical || result.clinical_reliability, ['copy', 'explanation'], demoResult.reliability.clinical.copy), quality: get(reliability.clinical || result.clinical_reliability, ['quality', 'quality_note'], demoResult.reliability.clinical.quality) } },
    fusion: { image: Number(get(fusion, ['image', 'image_weight', 'image_weight_wi'], demoResult.fusion.image)), clinical: Number(get(fusion, ['clinical', 'clinical_weight', 'clinical_weight_wc'], demoResult.fusion.clinical)), note: get(fusion, ['note', 'explanation', 'formula'], demoResult.fusion.note) },
    labels,
    reviewPriority: result.review_priority || 'routine',
    predictiveUncertainty: Number(result.predictive_uncertainty ?? actualTrust.uncertainty),
    uncertainty: result.uncertainty || {},
    retrieval: result.retrieval || {},
    reportComparison: result.report_comparison || {},
    explainability: result.explainability || {},
    guidance,
    heatmap: get(result, ['heatmap_note', 'attention_summary'], result.explainability?.message || demoResult.heatmap), explanation: get(result, ['clinical_explanation', 'explanation'], demoResult.explanation), evidence: result.evidence || result.retrieved_evidence || demoResult.evidence, audit: result.audit || result.audit_trail || demoResult.audit
  };
}

function percent(value) { const normalized = value > 1 ? value : value * 100; return `${Math.round(normalized)}%`; }
function setWidth(selector, value) { $(selector).style.width = `${Math.max(0, Math.min(100, value > 1 ? value : value * 100))}%`; }

function renderResult(data) {
  const trustStatus = String(data.trust.status).toLowerCase().includes('review') || String(data.trust.status).toLowerCase().includes('uncertain') ? 'review' : 'trusted';
  $('#finding-title').textContent = data.finding.title; $('#finding-summary').textContent = data.finding.summary;
  const labelItems = Array.isArray(data.labels) && data.labels.length ? data.labels : [{ label: data.finding.title, probability: data.trust.confidence, decision: 'abstain' }];
  $('#label-list').innerHTML = labelItems.map((item) => { const decision = String(item.decision || 'abstain'); const tone = decision.includes('positive') ? 'positive' : decision.includes('negative') ? 'negative' : 'abstain'; const text = decision.includes('positive') ? 'POSSIBLE · REVIEW' : decision.includes('negative') ? 'NO STRONG SIGNAL' : 'INSUFFICIENT SUPPORT'; return `<div class="label-row"><span class="label-name">${escapeHtml(plainLabel(item.label || 'Finding'))}</span><span class="label-prob">${percent(Number(item.probability || 0))}</span><span class="label-state ${tone}">${text}</span></div>`; }).join('');
  $('#trust-badge').className = `trust-badge ${trustStatus}`; $('#trust-badge .trust-icon').textContent = trustStatus === 'trusted' ? '✓' : '!'; $('#trust-badge strong').textContent = trustStatus === 'trusted' ? 'Support available' : 'Review needed'; $('#trust-badge small').textContent = 'Never a diagnosis';
  $('#confidence-value').textContent = percent(data.trust.confidence); setWidth('#confidence-meter', data.trust.confidence); $('#confidence-note').textContent = data.trust.note;
  $('#uncertainty-value').textContent = percent(data.trust.uncertainty); $('#uncertainty-marker').style.left = `${Math.max(0, Math.min(100, data.trust.uncertainty * 100))}%`;
  $('#review-priority').textContent = String(data.reviewPriority).toUpperCase(); $('#review-priority-note').textContent = data.uncertainty?.interpretation || 'Based on support and prediction spread';
  const image = data.reliability.image; const clinical = data.reliability.clinical;
  $('#image-reliability-score').textContent = percent(image.score); $('#image-reliability-title').textContent = image.label; $('#image-reliability-copy').textContent = image.copy; $('#image-quality-note').textContent = image.quality; setWidth('#image-reliability-meter', image.score);
  $('#clinical-reliability-score').textContent = percent(clinical.score); $('#clinical-reliability-title').textContent = clinical.label; $('#clinical-reliability-copy').textContent = clinical.copy; $('#clinical-quality-note').textContent = clinical.quality; setWidth('#clinical-reliability-meter', clinical.score);
  $('#image-weight').textContent = percent(data.fusion.image); $('#clinical-weight').textContent = percent(data.fusion.clinical); setWidth('#image-weight-bar', data.fusion.image); setWidth('#clinical-weight-bar', data.fusion.clinical); $('#fusion-note').textContent = data.fusion.note; $('#heatmap-note').textContent = data.heatmap; $('#clinical-explanation').textContent = data.explanation;
  renderGuidance(data.guidance || demoResult.guidance);
  const stage = $('.xray-stage'); const grid = data.explainability?.heatmap_grid; if (Array.isArray(grid) && grid.length) { let peak = { value: -1, row: 0, col: 0 }; grid.forEach((row, rowIndex) => row.forEach((value, colIndex) => { if (Number(value) > peak.value) peak = { value: Number(value), row: rowIndex, col: colIndex }; })); stage.classList.add('live-gradcam'); stage.style.setProperty('--cam-x', `${((peak.col + .5) / (grid[0]?.length || 8)) * 100}%`); stage.style.setProperty('--cam-y', `${((peak.row + .5) / grid.length) * 100}%`); }
  const evidence = Array.isArray(data.evidence) ? data.evidence : demoResult.evidence; $('#evidence-count').textContent = `${evidence.length} sources`; $('#retrieval-note').textContent = data.retrieval?.report_available ? `Local TF-IDF retrieval used the supplied MIMIC report (${data.retrieval.documents_considered || 0} documents considered).` : 'Local TF-IDF retrieval used citation metadata. Supply a study ID or permitted report text for report-grounded evidence.'; $('#evidence-list').innerHTML = evidence.map((item) => `<div class="evidence-item"><strong>${escapeHtml(item.title || item.name || 'Supporting source')}</strong><span class="citation">${escapeHtml(item.citation || item.reference || '[ref]')}</span><p>${escapeHtml(item.excerpt || item.source || item.citation_text || '')}</p></div>`).join('');
  const comparison = data.reportComparison || {}; $('#report-comparison').textContent = comparison.available ? `Report comparison available: ${comparison.comparison.filter((item) => item.report_state === 'mentioned_in_report').length} findings mentioned, ${comparison.comparison.filter((item) => item.report_state === 'negative_in_report').length} explicitly negative.` : comparison.message || 'No report comparison supplied.';
  const audit = Array.isArray(data.audit) ? data.audit : demoResult.audit; $('#audit-list').innerHTML = audit.map((item) => `<div class="audit-item"><time>${escapeHtml(item.time || item.timestamp || '—')}</time><p>${escapeHtml(item.text || item.event || '')}</p></div>`).join('');
  els.empty.hidden = true; els.content.hidden = false; $('#analysis-time').textContent = `Analyzed ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
}

const labelGlossary = {
  Atelectasis: 'Possible partial collapse of lung tissue', Cardiomegaly: 'Possible enlarged heart silhouette', Consolidation: 'Possible dense air-space change', Edema: 'Possible fluid-related lung change', 'Enlarged Cardiomediastinum': 'Possible widening near the heart and central chest', Fracture: 'Possible fracture-related finding', 'Lung Lesion': 'Possible focal lung lesion', 'Lung Opacity': 'Possible area of increased whiteness in the lung', 'No Finding': 'No strong abnormal finding signal', 'Pleural Effusion': 'Possible fluid around the lung', 'Pleural Other': 'Possible pleural-space finding', Pneumonia: 'Possible infection-like lung pattern', Pneumothorax: 'Possible air around the lung', 'Support Devices': 'Possible tube or line requiring position review'
};
function plainLabel(value) { return labelGlossary[value] || value; }

function renderGuidance(guidance) {
  const urgency = guidance?.urgency || demoResult.guidance.urgency;
  const treatment = guidance?.treatment || demoResult.guidance.treatment;
  const checklist = Array.isArray(guidance?.checklist) ? guidance.checklist : [];
  const tests = Array.isArray(guidance?.possible_tests) ? guidance.possible_tests : [];
  const reasons = Array.isArray(urgency.reasons) && urgency.reasons.length ? `Reason(s): ${urgency.reasons.join(', ')}.` : 'The AI result still requires independent clinician review.';
  $('#guidance-level').textContent = urgency.level === 'urgent' ? 'Urgent pathway' : urgency.level === 'prompt_review' ? 'Prompt review' : 'Routine review';
  $('#guidance-alert').className = `guidance-alert ${urgency.level || 'routine_review'}`;
  $('#guidance-alert-title').textContent = urgency.title || 'Clinician review required';
  $('#guidance-alert-message').textContent = urgency.message || '';
  $('#guidance-alert-reasons').textContent = reasons;
  $('#decision-checklist').innerHTML = checklist.map((item) => `<li>${escapeHtml(item)}</li>`).join('');
  $('#possible-tests').innerHTML = tests.map((item) => `<div class="test-prompt"><strong>${escapeHtml(item.title || 'Possible next step')}</strong><p>${escapeHtml(item.text || '')}</p></div>`).join('');
  $('#treatment-title').textContent = treatment.title || 'Medication and prescription safety';
  $('#treatment-status').textContent = treatment.status || 'No prescription generated';
  $('#treatment-message').textContent = treatment.message || 'No medication recommendation is generated by this tool.';
  $('#treatment-action').textContent = treatment.clinician_action || '';
  const language = guidance.model_language || {};
  $('#guidance-language').textContent = `${language.positive_signal || ''} ${language.negative_signal || ''} ${language.abstain || ''}`.trim();
  $('#guidance-references').innerHTML = (Array.isArray(guidance.references) ? guidance.references : []).map((item) => `<a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">${escapeHtml(item.title)}</a><span>${escapeHtml(item.note || '')}</span>`).join('');
}

function buildAnalyzePayload() {
  const clinical = formDataObject();
  if (selectedFile) return { image: selectedFile, clinical };
  return { clinical };
}

async function analyzeCase(event) {
  event.preventDefault();
  els.analyze.disabled = true; els.analyze.innerHTML = '<span class="button-icon">◌</span> Analyzing case…';
  try {
    const payload = buildAnalyzePayload();
    let response;
    const body = new FormData();
    if (payload.image) body.append('image', payload.image);
    body.append('clinical_data', JSON.stringify(payload.clinical));
    response = await fetchJson('/api/analyze', { method: 'POST', body });
    renderResult(normalizeResult(response)); showToast('Analysis complete. Review the confidence and reliability cards.');
  } catch (error) {
    renderResult(demoResult); showToast(`Backend unavailable — showing a demo readout. ${error.message}`);
  } finally { els.analyze.disabled = false; els.analyze.innerHTML = '<span class="button-icon">✦</span> Analyze case <span class="button-arrow">→</span>'; }
}
els.form.addEventListener('submit', analyzeCase);

async function submitFeedback(kind) {
  try {
    await fetchJson('/api/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ kind, case_filename: selectedFile?.name || null, displayed_at: new Date().toISOString() }) });
    showToast('Feedback saved locally for later analysis.');
  } catch (error) { showToast(`Feedback was not saved: ${error.message}`); }
}
document.querySelectorAll('[data-feedback]').forEach((button) => button.addEventListener('click', () => submitFeedback(button.dataset.feedback)));

async function loadSampleCase() {
  try {
    const sample = await fetchJson('/api/cases');
    const item = Array.isArray(sample) ? sample[0] : sample.cases?.[0] || sample.case || sample;
    if (item) {
      const fields = ['age', 'sex', 'view_position', 'symptoms', 'heart_rate', 'oxygen_saturation', 'temperature', 'notes'];
      fields.forEach((field) => { if (item[field] !== undefined && els.form.elements[field]) els.form.elements[field].value = item[field]; });
      setFile(item.image ? { name: item.image.name || 'sample-case.png', size: item.image.size || 1024 } : { name: 'sample-case.png', size: 186000 });
      showToast('Sample case loaded. Ready to analyze.'); return;
    }
  } catch (_) { /* Use the local sample when the API is unavailable. */ }
  els.form.elements.age.value = 68; els.form.elements.sex.value = 'Female'; els.form.elements.view_position.value = 'AP portable'; els.form.elements.symptoms.value = '3 days of dyspnea, productive cough, pleuritic chest pain'; els.form.elements.heart_rate.value = 104; els.form.elements.oxygen_saturation.value = 94; els.form.elements.temperature.value = 38.2; els.form.elements.notes.value = 'No prior study available for comparison.'; setFile({ name: 'sample-case.png', size: 186000 }); showToast('Sample case loaded. Ready to analyze.');
}
$('#sample-case').addEventListener('click', loadSampleCase);

async function loadStats() {
  try { const stats = await fetchJson('/api/stats'); const sync = stats.last_sync || stats.updated_at || stats.last_updated; if (sync) $('#last-sync').textContent = new Date(sync).toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' }); $('#model-status').textContent = stats.model?.mode === 'densenet121_trained' ? 'DenseNet-121' : stats.model?.mode === 'trained_baseline' ? 'Patient-level baseline' : 'Demo-safe prototype'; $('#dataset-count').textContent = `${stats.images ?? 0} unique images`; $('#data-integrity').textContent = `${stats.duplicate_image_files_excluded ?? 0} duplicate files excluded`; const metrics = await fetchJson('/api/metrics'); const test = metrics.aggregate?.test; if (test?.auroc != null) $('#model-metric').textContent = `Held-out AUROC ${(test.auroc * 100).toFixed(1)}% · F1 ${(test.f1 * 100).toFixed(1)}%`; } catch (_) { $('#last-sync').textContent = 'Local only'; $('#model-metric').textContent = 'Metrics unavailable'; $('#dataset-count').textContent = 'Local only'; $('#data-integrity').textContent = 'Integrity status unavailable'; }
}

async function loadComparison() {
  try {
    const comparison = await fetchJson('/api/comparison');
    if (!comparison.available) throw new Error('comparison unavailable');
    const selection = comparison.selection || {};
    $('#selection-summary').textContent = `${selection.deployed_image_model || 'image'} · ${selection.deployed_clinical_model || 'clinical'} · ${selection.deployed_fusion || 'fusion'}`;
    const rows = [...(comparison.image_candidates || []), ...(comparison.clinical_candidates || []), ...(comparison.fusion_candidates || [])];
    const metric = (row, key) => row.metrics?.[key] == null ? '—' : (Number(row.metrics[key]) * 100).toFixed(1) + '%';
    $('#model-comparison-table').innerHTML = `<table class="model-table"><thead><tr><th>Candidate</th><th>AUROC</th><th>AUPRC</th><th>F1</th><th>Brier</th><th>ECE</th><th>Time</th></tr></thead><tbody>${rows.map((row) => { const selected = [selection.deployed_image_model, selection.deployed_clinical_model, selection.deployed_fusion].includes(row.name); return `<tr class="${selected ? 'selected' : ''}"><td>${escapeHtml(row.name)}${selected ? ' · selected' : ''}</td><td>${metric(row,'auroc')}</td><td>${metric(row,'auprc')}</td><td>${metric(row,'f1')}</td><td>${metric(row,'brier_score')}</td><td>${metric(row,'ece_10bin')}</td><td>${Number(row.training_seconds || 0).toFixed(1)} s</td></tr>`; }).join('')}</tbody></table>`;
    const charts = ['model_comparison_metrics.png','per_label_auroc_heatmap.png','selected_confusion_heatmap.png','clinical_feature_importance.png','calibration_comparison.png','performance_vs_compute.png','cohort_split_pie.png','label_distribution.png','training_curve.png'];
    $('#research-charts').innerHTML = charts.map((name) => `<figure><img loading="lazy" src="/artifacts/comparison/${name}" alt="${escapeHtml(name)}" /><figcaption>${escapeHtml(name.replace('.png','').replaceAll('_',' '))}</figcaption></figure>`).join('');
  } catch (_) { $('#selection-summary').textContent = 'Comparison not generated'; $('#research-note').textContent = 'Run the comparative study script to populate model comparisons and research figures.'; }
}
loadStats();
loadComparison();
