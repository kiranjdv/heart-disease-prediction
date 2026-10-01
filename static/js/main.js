/**
 * main.js - ProHealth Cardiology UI Interactions & Live AJAX Engine
 */

document.addEventListener('DOMContentLoaded', () => {
  // Modal handlers
  const editDetailsBtn = document.getElementById('editDetailsBtn');
  const diagnosticModal = document.getElementById('diagnosticModal');
  const closeModalBtn = document.getElementById('closeModalBtn');
  const diagnosticForm = document.getElementById('diagnosticForm');

  if (editDetailsBtn && diagnosticModal) {
    editDetailsBtn.addEventListener('click', () => {
      diagnosticModal.style.display = 'flex';
    });
  }

  if (closeModalBtn && diagnosticModal) {
    closeModalBtn.addEventListener('click', () => {
      diagnosticModal.style.display = 'none';
    });
  }

  if (diagnosticModal) {
    window.addEventListener('click', (e) => {
      if (e.target === diagnosticModal) {
        diagnosticModal.style.display = 'none';
      }
    });
  }

  // Live AJAX Diagnostic Form Submission
  if (diagnosticForm) {
    diagnosticForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const formData = new FormData(diagnosticForm);
      const payload = {};
      formData.forEach((value, key) => {
        payload[key] = parseFloat(value);
      });

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const result = await response.json();

        if (result.success) {
          // Update live floating callout chips on the 3D Heart
          const bpEl = document.getElementById('floatingBpVal');
          const hrEl = document.getElementById('floatingHrVal');
          const riskEl = document.getElementById('floatingRiskVal');
          const riskBadge = document.getElementById('floatingRiskBadge');

          if (bpEl && payload.trestbps) bpEl.innerText = `${payload.trestbps}/80`;
          if (hrEl && payload.thalach) hrEl.innerText = `${payload.thalach}`;
          if (riskEl) riskEl.innerText = `${result.probability_percent}% Probability`;

          if (riskBadge) {
            riskBadge.innerText = result.risk_label;
            riskBadge.className = `status-chip ${result.prediction === 1 ? 'status-high' : 'status-ok'}`;
          }

          // Update Diagnosis Note in the Patient Card
          const diagNote = document.getElementById('patientDiagNote');
          if (diagNote) {
            diagNote.innerText = result.prediction === 1 ? 'Elevated Cardiac Risk' : 'Normal Cardiac Function';
            diagNote.style.color = result.prediction === 1 ? '#EF4444' : '#15803D';
          }

          // Update Demographics
          const demoEl = document.getElementById('patientDemographicsVal');
          if (demoEl && payload.age) {
            const sexLabel = payload.sex === 1 ? 'Male' : 'Female';
            demoEl.innerText = `${payload.age} yrs • ${sexLabel}`;
          }

          // Update Cholesterol Biomarker Tile
          const cholValEl = document.getElementById('biomarkerCholVal');
          const cholBadgeEl = document.getElementById('cholStatusBadge');
          const cholSubEl = document.getElementById('cholSubtext');
          if (cholValEl && payload.chol !== undefined) {
            cholValEl.innerHTML = `${payload.chol} <span style="font-size: 0.85rem; font-weight: 600; color: #64748B;">mg/dl</span>`;
            if (cholBadgeEl) {
              if (payload.chol < 200) {
                cholBadgeEl.innerText = 'Normal';
                cholBadgeEl.className = 'status-chip status-ok';
              } else if (payload.chol < 240) {
                cholBadgeEl.innerText = 'Borderline';
                cholBadgeEl.className = 'status-chip status-elevated';
              } else {
                cholBadgeEl.innerText = 'High';
                cholBadgeEl.className = 'status-chip status-high';
              }
            }
            if (cholSubEl) {
              cholSubEl.innerText = payload.chol < 200 ? 'Optimal target (<200)' : (payload.chol < 240 ? 'Borderline high (200–239)' : 'Hypercholesterolemia (≥240)');
            }
          }

          // Update ST Depression Biomarker Tile
          const stValEl = document.getElementById('biomarkerStVal');
          const stBadgeEl = document.getElementById('stStatusBadge');
          const stSubEl = document.getElementById('stSubtext');
          if (stValEl && payload.oldpeak !== undefined) {
            stValEl.innerHTML = `${payload.oldpeak} <span style="font-size: 0.85rem; font-weight: 600; color: #64748B;">mm</span>`;
            if (stBadgeEl) {
              if (payload.oldpeak < 1.0) {
                stBadgeEl.innerText = 'Normal';
                stBadgeEl.className = 'status-chip status-ok';
              } else if (payload.oldpeak < 2.0) {
                stBadgeEl.innerText = 'Mild';
                stBadgeEl.className = 'status-chip status-elevated';
              } else {
                stBadgeEl.innerText = 'Ischemic';
                stBadgeEl.className = 'status-chip status-high';
              }
            }
            if (stSubEl) {
              const caVal = payload.ca || 0;
              stSubEl.innerText = caVal > 0 ? `${caVal} major vessel(s) obstructed` : 'No vessel blockage (ca=0)';
            }
          }

          // Update AI Clinical Protocol Card
          const protocolBadge = document.getElementById('protocolRiskBadge');
          if (protocolBadge) {
            protocolBadge.innerText = result.prediction === 1 ? 'High Risk Care Pathway' : 'Preventive Care Pathway';
            protocolBadge.className = `status-chip ${result.prediction === 1 ? 'status-high' : 'status-ok'}`;
          }

          const protocolList = document.getElementById('protocolList');
          if (protocolList && result.recommendations && result.recommendations.length) {
            protocolList.innerHTML = '';
            result.recommendations.forEach((rec) => {
              const row = document.createElement('div');
              row.className = `schedule-row ${rec.type}`;

              let iconSvg = '';
              if (rec.icon === 'alert') {
                iconSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;
              } else if (rec.icon === 'pill') {
                iconSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#D97706" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m10.5 4.5 9 9a4.95 4.95 0 1 1-7 7l-9-9a4.95 4.95 0 1 1 7-7Z"></path><line x1="8.5" y1="8.5" x2="15.5" y2="15.5"></line></svg>`;
              } else if (rec.icon === 'activity') {
                iconSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7C3AED" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>`;
              } else if (rec.icon === 'calendar') {
                iconSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>`;
              } else if (rec.icon === 'heart') {
                iconSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"></path></svg>`;
              } else {
                iconSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0284C7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
              }

              row.innerHTML = `
                <div>
                  <div class="sched-title">
                    ${iconSvg}
                    ${rec.title}
                  </div>
                  <div class="sched-sub">${rec.sub}</div>
                </div>
                <div class="sched-time-badge ${rec.type}">${rec.badge}</div>
              `;
              protocolList.appendChild(row);
            });
          }

          // Close modal and show notification
          if (diagnosticModal) diagnosticModal.style.display = 'none';
        }
      } catch (err) {
        console.error('Inference error:', err);
      }
    });
  }

  // Batch CSV Upload Handler
  const batchUploadForm = document.getElementById('batchUploadForm');
  if (batchUploadForm) {
    batchUploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const fileInput = document.getElementById('csvFileInput');
      if (!fileInput.files.length) return;

      const formData = new FormData();
      formData.append('file', fileInput.files[0]);

      const submitBtn = batchUploadForm.querySelector('button[type="submit"]');
      if (submitBtn) submitBtn.innerText = 'Analyzing Cohort...';

      try {
        const response = await fetch('/api/batch-process', {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();

        if (data.success) {
          // Render results summary
          document.getElementById('batchStats').style.display = 'grid';
          document.getElementById('totalCount').innerText = data.total;
          document.getElementById('highRiskCount').innerText = data.high_risk;
          document.getElementById('lowRiskCount').innerText = data.low_risk;
          document.getElementById('avgRisk').innerText = `${data.avg_risk}%`;

          // Render preview table
          const tbody = document.getElementById('batchTableBody');
          tbody.innerHTML = '';
          data.preview.forEach((row) => {
            const tr = document.createElement('tr');
            const isHigh = row.predicted_risk === 'High Risk';
            tr.innerHTML = `
              <td>${row.age} yo (${row.sex === 1 ? 'M' : 'F'})</td>
              <td>${row.trestbps} mmHg</td>
              <td>${row.chol} mg/dl</td>
              <td>${row.thalach} bpm</td>
              <td>${row.oldpeak}</td>
              <td>${row.ca}</td>
              <td><span class="status-chip ${isHigh ? 'status-high' : 'status-ok'}">${row.predicted_risk}</span></td>
              <td><b>${row['risk_probability_%']}%</b></td>
            `;
            tbody.appendChild(tr);
          });
          document.getElementById('batchResultsTable').style.display = 'block';
        } else {
          alert('Error processing file: ' + (data.error || 'Unknown error'));
        }
      } catch (err) {
        alert('Server connection error during batch processing.');
      } finally {
        if (submitBtn) submitBtn.innerText = 'Upload & Screen Cohort';
      }
    });
  }

  // Initialize features
  initWhatIfSimulator();
  initHeaderControls();
});

// ==========================================
// AI WHAT-IF RISK SIMULATOR
// ==========================================
function initWhatIfSimulator() {
  const bpSlider = document.getElementById('whatif_bp');
  const cholSlider = document.getElementById('whatif_chol');
  const hrSlider = document.getElementById('whatif_hr');
  if (!bpSlider) return;

  const bpVal = document.getElementById('whatif_bp_val');
  const cholVal = document.getElementById('whatif_chol_val');
  const hrVal = document.getElementById('whatif_hr_val');

  const simRiskProb = document.getElementById('simulatedRiskProb');
  const simRiskBadge = document.getElementById('simulatedRiskBadge');
  const simDeltaBadge = document.getElementById('simulatedDeltaBadge');

  let debounceTimer;

  async function recalculateSimulated() {
    const targetBp = parseFloat(bpSlider.value);
    const targetChol = parseFloat(cholSlider.value);
    const targetHr = parseFloat(hrSlider.value);

    if (bpVal) bpVal.innerText = `${targetBp} mmHg`;
    if (cholVal) cholVal.innerText = `${targetChol} mg/dl`;
    if (hrVal) hrVal.innerText = `${targetHr} bpm`;

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
      const basePayload = {
        age: 55,
        sex: 1,
        cp: 1,
        trestbps: targetBp,
        chol: targetChol,
        fbs: targetChol > 220 ? 1 : 0,
        restecg: 0,
        thalach: targetHr,
        exang: targetHr < 110 ? 1 : 0,
        oldpeak: targetBp > 150 ? 1.8 : 0.4,
        slope: targetBp > 150 ? 1 : 2,
        ca: targetChol > 260 ? 1 : 0,
        thal: 2,
      };

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(basePayload),
        });
        const data = await res.json();
        if (data.success) {
          if (simRiskProb) {
            simRiskProb.innerText = `${data.probability_percent}%`;
            simRiskProb.style.color = data.prediction === 1 ? '#EF4444' : '#10B981';
          }
          if (simRiskBadge) {
            simRiskBadge.innerText = data.risk_label.toUpperCase();
            simRiskBadge.className = `status-chip ${data.prediction === 1 ? 'status-high' : 'status-ok'}`;
          }

          const baseRisk = 85.7;
          const delta = (baseRisk - data.probability_percent).toFixed(1);
          if (simDeltaBadge) {
            if (delta > 0) {
              simDeltaBadge.innerText = `-${delta}% Lower Risk (ARR)`;
              simDeltaBadge.style.background = '#DCFCE7';
              simDeltaBadge.style.color = '#15803D';
            } else {
              simDeltaBadge.innerText = `+${Math.abs(delta)}% Higher Risk`;
              simDeltaBadge.style.background = '#FEE2E2';
              simDeltaBadge.style.color = '#B91C1C';
            }
          }
        }
      } catch (err) {
        console.error('Simulation error:', err);
      }
    }, 120);
  }

  bpSlider.addEventListener('input', recalculateSimulated);
  cholSlider.addEventListener('input', recalculateSimulated);
  hrSlider.addEventListener('input', recalculateSimulated);

  recalculateSimulated();
}

// ==========================================
// SPOTLIGHT PATIENT SEARCH & HEADER CONTROLS
// ==========================================
function initHeaderControls() {
  const searchBtn = document.getElementById('searchRecordsBtn');
  const searchModal = document.getElementById('patientSearchModal');
  const closeSearchBtn = document.getElementById('closeSearchModalBtn');
  const searchInput = document.getElementById('patientSearchInput');
  const resultsContainer = document.getElementById('patientSearchResults');
  const filterChips = document.querySelectorAll('.search-filter-chip');

  const profileBtn = document.getElementById('userProfileBtn');
  const profileDropdown = document.getElementById('profileDropdown');

  // Toggle physician profile popover
  if (profileBtn && profileDropdown) {
    profileBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      profileDropdown.classList.toggle('active');
    });
  }

  // Close popovers on click outside
  document.addEventListener('click', (e) => {
    if (profileDropdown && !profileDropdown.contains(e.target) && e.target !== profileBtn) {
      profileDropdown.classList.remove('active');
    }
  });

  // Open Search Modal
  function openSearchModal() {
    if (!searchModal) return;
    searchModal.style.display = 'flex';
    if (searchInput) {
      searchInput.value = '';
      setTimeout(() => searchInput.focus(), 50);
    }
    fetchSearchResults('', 'all');
  }

  function closeSearchModal() {
    if (searchModal) searchModal.style.display = 'none';
  }

  if (searchBtn) searchBtn.addEventListener('click', openSearchModal);
  if (closeSearchBtn) closeSearchBtn.addEventListener('click', closeSearchModal);

  if (searchModal) {
    searchModal.addEventListener('click', (e) => {
      if (e.target === searchModal) closeSearchModal();
    });
  }

  // Shortcut Ctrl+K or / to open search
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey && e.key.toLowerCase() === 'k') || (e.key === '/' && document.activeElement.tagName !== 'INPUT')) {
      e.preventDefault();
      openSearchModal();
    }
    if (e.key === 'Escape') {
      closeSearchModal();
      if (profileDropdown) profileDropdown.classList.remove('active');
    }
  });

  let currentFilter = 'all';
  filterChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      filterChips.forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');
      currentFilter = chip.getAttribute('data-filter') || 'all';
      fetchSearchResults(searchInput ? searchInput.value : '', currentFilter);
    });
  });

  let searchTimer;
  if (searchInput) {
    searchInput.addEventListener('input', () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => {
        fetchSearchResults(searchInput.value, currentFilter);
      }, 150);
    });
  }

  async function fetchSearchResults(query, filter) {
    if (!resultsContainer) return;
    resultsContainer.innerHTML = '<div style="padding: 18px; text-align: center; color: #94A3B8; font-size: 0.85rem;">Searching cohort database...</div>';

    try {
      const res = await fetch(`/api/patients/search?q=${encodeURIComponent(query)}&filter=${encodeURIComponent(filter)}`);
      const data = await res.json();
      const results = data.results || [];

      if (!results.length) {
        resultsContainer.innerHTML = '<div style="padding: 24px; text-align: center; color: #94A3B8; font-size: 0.88rem;">No matching patient records found in dataset. Try searching by ID (e.g. <code>PID-010</code>), Name, or Age.</div>';
        return;
      }

      resultsContainer.innerHTML = '';
      results.forEach((p) => {
        const item = document.createElement('a');
        item.href = p.url;
        item.className = 'search-result-item';

        const isHigh = p.is_high_risk;
        const tag = p.is_preset ? 'Preset Profile' : (isHigh ? 'High Risk' : 'Low Risk');
        const tagClass = isHigh ? 'status-high' : 'status-ok';

        item.innerHTML = `
          <div class="search-item-left">
            <span class="pid-pill">${p.pid}</span>
            <div>
              <div class="search-item-title">${p.name}</div>
              <div class="search-item-meta">${p.age} yrs • ${p.sex} • BP: ${p.trestbps} mmHg • Chol: ${p.chol} mg/dl</div>
            </div>
          </div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span class="status-chip ${tagClass}" style="font-size: 0.72rem;">${tag}</span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M9 18l6-6-6-6"/>
            </svg>
          </div>
        `;
        resultsContainer.appendChild(item);
      });
    } catch (err) {
      resultsContainer.innerHTML = '<div style="padding: 16px; text-align: center; color: #EF4444; font-size: 0.85rem;">Error querying patient dataset.</div>';
    }
  }
}

