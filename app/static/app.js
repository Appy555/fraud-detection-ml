/**
 * FraudGuard AI - Frontend Controller & Interaction Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const apiStatusPill = document.getElementById('apiStatusPill');
  const apiStatusText = document.getElementById('apiStatusText');
  const navThresholdValue = document.getElementById('navThresholdValue');
  const latencyStat = document.getElementById('latencyStat');

  const fraudForm = document.getElementById('fraudForm');
  const inputAmount = document.getElementById('inputAmount');
  const inputTime = document.getElementById('inputTime');
  const thresholdSlider = document.getElementById('thresholdSlider');
  const sliderValueDisplay = document.getElementById('sliderValueDisplay');
  const thresholdIndicatorLabel = document.getElementById('thresholdIndicatorLabel');
  const thresholdMarker = document.getElementById('thresholdMarker');

  const pcaInputsContainer = document.getElementById('pcaInputsContainer');
  const btnSubmitPredict = document.getElementById('btnSubmitPredict');
  const btnSpinner = document.getElementById('btnSpinner');

  const resultPlaceholder = document.getElementById('resultPlaceholder');
  const resultContent = document.getElementById('resultContent');
  const verdictBanner = document.getElementById('verdictBanner');
  const verdictIcon = document.getElementById('verdictIcon');
  const verdictStatusText = document.getElementById('verdictStatusText');
  const verdictRiskTag = document.getElementById('verdictRiskTag');
  const meterFillCircle = document.getElementById('meterFillCircle');
  const meterPercentageText = document.getElementById('meterPercentageText');
  const riskBarFill = document.getElementById('riskBarFill');
  const detailRawProb = document.getElementById('detailRawProb');
  const detailThreshold = document.getElementById('detailThreshold');
  const detailAmount = document.getElementById('detailAmount');
  const detailAction = document.getElementById('detailAction');
  const vectorTagsContainer = document.getElementById('vectorTagsContainer');

  const auditTableBody = document.getElementById('auditTableBody');
  const btnClearHistory = document.getElementById('btnClearHistory');

  // Preset Buttons
  const btnPresetLegit = document.getElementById('btnPresetLegit');
  const btnPresetFraud = document.getElementById('btnPresetFraud');
  const btnPresetSuspicious = document.getElementById('btnPresetSuspicious');
  const btnPresetRandom = document.getElementById('btnPresetRandom');
  const btnZeroPca = document.getElementById('btnZeroPca');
  const btnJitterPca = document.getElementById('btnJitterPca');

  let currentRawProbability = null;
  let serverDefaultThreshold = 0.99;
  const auditHistory = [];

  // ==========================================
  // 1. Initialize PCA Inputs (V1 to V28)
  // ==========================================
  function initializePcaInputs() {
    pcaInputsContainer.innerHTML = '';
    for (let i = 1; i <= 28; i++) {
      const item = document.createElement('div');
      item.className = 'pca-item';
      item.innerHTML = `
        <label class="pca-item-label" for="inputV${i}">V${i}</label>
        <input type="number" step="0.0001" id="inputV${i}" data-v="${i}" value="0.0000" class="custom-input pca-item-input" />
      `;
      pcaInputsContainer.appendChild(item);
    }
  }

  // Helper to get all PCA vector values
  function getPcaValues() {
    const values = {};
    for (let i = 1; i <= 28; i++) {
      const el = document.getElementById(`inputV${i}`);
      values[`V${i}`] = el ? parseFloat(el.value) || 0.0 : 0.0;
    }
    return values;
  }

  // Helper to set all PCA vector values
  function setPcaValues(values) {
    for (let i = 1; i <= 28; i++) {
      const el = document.getElementById(`inputV${i}`);
      if (el) {
        el.value = (values[`V${i}`] !== undefined ? values[`V${i}`] : 0.0).toFixed(4);
      }
    }
  }

  // ==========================================
  // 2. Health Check & Initial Setup
  // ==========================================
  async function checkApiHealth() {
    const startTime = performance.now();
    try {
      const response = await fetch('/api/health');
      const endTime = performance.now();
      const latency = Math.round(endTime - startTime);

      if (response.ok) {
        const data = await response.json();
        serverDefaultThreshold = data.threshold || 0.99;
        navThresholdValue.textContent = serverDefaultThreshold.toFixed(2);
        thresholdSlider.value = serverDefaultThreshold;
        updateSliderDisplay(serverDefaultThreshold);
        latencyStat.textContent = `${latency} ms`;

        apiStatusPill.className = 'status-pill';
        apiStatusText.textContent = 'API Connected';
      } else {
        throw new Error('API returned unhealthy status');
      }
    } catch (err) {
      console.warn('API health check failed:', err);
      apiStatusPill.className = 'status-pill error';
      apiStatusText.textContent = 'API Offline';
    }
  }

  // ==========================================
  // 3. Slider Threshold Interaction
  // ==========================================
  function updateSliderDisplay(val) {
    const num = parseFloat(val);
    sliderValueDisplay.textContent = num.toFixed(2);
    thresholdIndicatorLabel.textContent = `Threshold ${(num * 100).toFixed(1)}%`;
    thresholdMarker.style.left = `${num * 100}%`;

    // If a prediction is already visible, re-render verdict based on active threshold
    if (currentRawProbability !== null) {
      renderVerdict(currentRawProbability, num, parseFloat(inputAmount.value) || 0);
    }
  }

  thresholdSlider.addEventListener('input', (e) => {
    updateSliderDisplay(e.target.value);
  });

  // ==========================================
  // 4. Presets & Scenarios
  // ==========================================
  const PRESETS = {
    legit: {
      amount: 25.40,
      time: 512.0,
      pca: {
        V1: 0.12, V2: -0.05, V3: 0.45, V4: -0.18, V5: 0.32, V6: -0.22,
        V7: 0.11, V8: 0.04, V9: -0.08, V10: 0.05, V11: -0.15, V12: 0.21,
        V13: -0.31, V14: 0.08, V15: 0.52, V16: -0.11, V17: 0.03, V18: -0.02,
        V19: 0.14, V20: -0.04, V21: 0.01, V22: -0.03, V23: 0.02, V24: 0.05,
        V25: -0.01, V26: 0.04, V27: 0.01, V28: -0.01
      }
    },
    fraud: {
      amount: 1499.00,
      time: 406.0,
      pca: {
        V1: -2.31, V2: 1.95, V3: -1.61, V4: 4.82, V5: -0.52, V6: -1.43,
        V7: -2.53, V8: 1.39, V9: -2.77, V10: -4.56, V11: 3.88, V12: -6.14,
        V13: -0.32, V14: -7.21, V15: 0.12, V16: -4.18, V17: -5.32, V18: -1.82,
        V19: 0.45, V20: 0.38, V21: 0.52, V22: -0.03, V23: -0.47, V24: 0.32,
        V25: 0.04, V26: 0.17, V27: 0.26, V28: -0.14
      }
    },
    suspicious: {
      amount: 480.00,
      time: 1250.0,
      pca: {
        V1: -1.05, V2: 0.82, V3: -0.74, V4: 1.92, V5: -0.31, V6: -0.62,
        V7: -1.15, V8: 0.44, V9: -1.22, V10: -1.95, V11: 1.45, V12: -2.35,
        V13: 0.12, V14: -2.85, V15: 0.32, V16: -1.45, V17: -1.88, V18: -0.72,
        V19: 0.25, V20: 0.18, V21: 0.22, V22: -0.08, V23: -0.15, V24: 0.11,
        V25: 0.08, V26: -0.04, V27: 0.12, V28: -0.05
      }
    }
  };

  function applyPreset(presetKey) {
    const p = PRESETS[presetKey];
    if (!p) return;
    inputAmount.value = p.amount.toFixed(2);
    inputTime.value = p.time.toFixed(1);
    setPcaValues(p.pca);

    // Auto-trigger prediction for immediate visual feedback
    fraudForm.dispatchEvent(new Event('submit'));
  }

  btnPresetLegit.addEventListener('click', () => applyPreset('legit'));
  btnPresetFraud.addEventListener('click', () => applyPreset('fraud'));
  btnPresetSuspicious.addEventListener('click', () => applyPreset('suspicious'));

  btnPresetRandom.addEventListener('click', () => {
    inputAmount.value = (Math.random() * 800 + 5).toFixed(2);
    inputTime.value = Math.floor(Math.random() * 100000);
    const randomPca = {};
    for (let i = 1; i <= 28; i++) {
      randomPca[`V${i}`] = (Math.random() * 4 - 2);
    }
    setPcaValues(randomPca);
    fraudForm.dispatchEvent(new Event('submit'));
  });

  btnZeroPca.addEventListener('click', () => {
    const zeros = {};
    for (let i = 1; i <= 28; i++) zeros[`V${i}`] = 0.0;
    setPcaValues(zeros);
  });

  btnJitterPca.addEventListener('click', () => {
    for (let i = 1; i <= 28; i++) {
      const el = document.getElementById(`inputV${i}`);
      if (el) {
        const val = parseFloat(el.value) || 0.0;
        el.value = (val + (Math.random() * 0.4 - 0.2)).toFixed(4);
      }
    }
  });

  // ==========================================
  // 5. Prediction Execution
  // ==========================================
  fraudForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const amount = parseFloat(inputAmount.value) || 0.0;
    const time = parseFloat(inputTime.value) || 0.0;
    const pcaData = getPcaValues();
    const activeThreshold = parseFloat(thresholdSlider.value) || 0.99;

    const payload = {
      Time: time,
      ...pcaData,
      Amount: amount
    };

    // UI Loading state
    btnSpinner.classList.remove('hidden');
    btnSubmitPredict.disabled = true;
    const startTime = performance.now();

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const latency = Math.round(performance.now() - startTime);
      latencyStat.textContent = `${latency} ms`;

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to score transaction');
      }

      const result = await response.json();
      currentRawProbability = result.fraud_probability;

      // Render Results HUD
      renderVerdict(currentRawProbability, activeThreshold, amount);
      renderAnomalyVectors(pcaData);

      // Add to Session Audit History
      addToAuditHistory({
        timestamp: new Date().toLocaleTimeString(),
        amount: amount,
        time: time,
        probability: currentRawProbability,
        threshold: activeThreshold,
        isFraud: currentRawProbability >= activeThreshold
      });

    } catch (err) {
      alert(`Error predicting transaction: ${err.message}`);
    } finally {
      btnSpinner.classList.add('hidden');
      btnSubmitPredict.disabled = false;
    }
  });

  // ==========================================
  // 6. Verdict & Gauge Rendering
  // ==========================================
  function renderVerdict(prob, threshold, amount) {
    resultPlaceholder.classList.add('hidden');
    resultContent.classList.remove('hidden');

    const isFraud = prob >= threshold;
    const probPercent = (prob * 100).toFixed(2);

    // Update Radial Gauge Circle
    // circumference = 2 * PI * 70 ≈ 440
    const circumference = 440;
    const offset = circumference - (prob * circumference);
    meterFillCircle.style.strokeDashoffset = offset;
    meterPercentageText.textContent = `${probPercent}%`;

    // Meter Fill Color
    if (isFraud) {
      meterFillCircle.style.stroke = 'var(--status-red)';
    } else if (prob >= 0.50) {
      meterFillCircle.style.stroke = 'var(--status-amber)';
    } else {
      meterFillCircle.style.stroke = 'var(--status-green)';
    }

    // Update Progress Bar
    riskBarFill.style.width = `${Math.min(prob * 100, 100)}%`;

    // Verdict Banner Styling
    if (isFraud) {
      verdictBanner.className = 'verdict-banner fraud';
      verdictIcon.innerHTML = `
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="15" y1="9" x2="9" y2="15"></line>
          <line x1="9" y1="9" x2="15" y2="15"></line>
        </svg>
      `;
      verdictStatusText.textContent = 'HIGH RISK FRAUD DETECTED';
      verdictRiskTag.textContent = 'TRANSACTION BLOCKED';
      detailAction.textContent = 'DECLINE & ALERT FRAUD OPS';
      detailAction.style.color = '#f87171';
    } else {
      verdictBanner.className = 'verdict-banner legit';
      verdictIcon.innerHTML = `
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
          <polyline points="22 4 12 14.01 9 11.01"></polyline>
        </svg>
      `;
      verdictStatusText.textContent = 'LEGITIMATE TRANSACTION';
      verdictRiskTag.textContent = 'SAFE / APPROVED';
      detailAction.textContent = 'AUTHORIZE PAYMENT';
      detailAction.style.color = '#34d399';
    }

    // Detail Tiles
    detailRawProb.textContent = prob.toFixed(6);
    detailThreshold.textContent = threshold.toFixed(4);
    detailAmount.textContent = `$${amount.toFixed(2)}`;
  }

  // ==========================================
  // 7. Top Anomaly Feature Badges
  // ==========================================
  function renderAnomalyVectors(pcaData) {
    vectorTagsContainer.innerHTML = '';
    
    // Sort features by absolute magnitude deviation
    const sorted = Object.entries(pcaData)
      .map(([k, v]) => ({ name: k, val: v, abs: Math.abs(v) }))
      .sort((a, b) => b.abs - a.abs)
      .slice(0, 6);

    sorted.forEach(f => {
      const tag = document.createElement('span');
      const isExtreme = f.abs >= 2.0;
      tag.className = `vector-tag ${isExtreme ? 'anomaly' : ''}`;
      tag.textContent = `${f.name}: ${f.val >= 0 ? '+' : ''}${f.val.toFixed(2)}`;
      vectorTagsContainer.appendChild(tag);
    });
  }

  // ==========================================
  // 8. Session Audit History
  // ==========================================
  function addToAuditHistory(item) {
    auditHistory.unshift(item);
    renderAuditTable();
  }

  function renderAuditTable() {
    if (auditHistory.length === 0) {
      auditTableBody.innerHTML = `
        <tr id="emptyAuditRow">
          <td colspan="8" class="empty-table-msg">No transactions analyzed yet. Try submitting one above!</td>
        </tr>
      `;
      return;
    }

    auditTableBody.innerHTML = '';
    auditHistory.slice(0, 20).forEach((entry) => {
      const tr = document.createElement('tr');
      const isFraud = entry.isFraud;
      tr.innerHTML = `
        <td>${entry.timestamp}</td>
        <td><strong>$${entry.amount.toFixed(2)}</strong></td>
        <td>${entry.time.toFixed(0)}s</td>
        <td><span class="mono">${(entry.probability * 100).toFixed(2)}%</span></td>
        <td><span class="mono">${entry.threshold.toFixed(2)}</span></td>
        <td>
          <span class="audit-status-badge ${isFraud ? 'fraud' : 'legit'}">
            ${isFraud ? 'FRAUD' : 'APPROVED'}
          </span>
        </td>
        <td>${isFraud ? 'CRITICAL' : (entry.probability > 0.5 ? 'MEDIUM' : 'LOW')}</td>
        <td>
          <span style="color: ${isFraud ? '#f87171' : '#34d399'}; font-weight: 600;">
            ${isFraud ? 'BLOCKED' : 'PROCESSED'}
          </span>
        </td>
      `;
      auditTableBody.appendChild(tr);
    });
  }

  btnClearHistory.addEventListener('click', () => {
    auditHistory.length = 0;
    renderAuditTable();
  });

  // Initialize
  initializePcaInputs();
  checkApiHealth();
});
