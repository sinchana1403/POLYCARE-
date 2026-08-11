'use strict';

const API_BASE =
  (window.location.hostname === 'localhost' ||
   window.location.hostname === '127.0.0.1')
  ? 'http://127.0.0.1:5000/api'
  : '/api';

const STATE = {
  lastResult: null,
  darkMode:   true,
  charts:     {},
};

const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function setCookie(name, val, days = 365) {
  document.cookie = `${name}=${val};path=/;max-age=${days*86400}`;
}
function getCookie(name) {
  const m = document.cookie.match(new RegExp(`(^| )${name}=([^;]+)`));
  return m ? m[2] : null;
}

function initDarkMode() {
  const stored = getCookie('theme');
  STATE.darkMode = stored !== 'light';
  applyTheme();
}
function applyTheme() {
  document.body.classList.toggle('light-mode', !STATE.darkMode);
  $$('.dark-toggle').forEach(btn => {
    btn.textContent = STATE.darkMode ? '☀️' : '🌙';
    btn.title = STATE.darkMode ? 'Switch to light mode' : 'Switch to dark mode';
  });
  setCookie('theme', STATE.darkMode ? 'dark' : 'light');
}
function toggleDarkMode() {
  STATE.darkMode = !STATE.darkMode;
  applyTheme();
}

function showSpinner(msg = 'Analysing…', sub = 'Please wait') {
  let overlay = $('#spinner-overlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'spinner-overlay';
    overlay.className = 'spinner-overlay';
    overlay.innerHTML = `
      <div class="spinner-ring"></div>
      <div class="spinner-inner">
        <p class="spinner-text" id="spinner-msg">${msg}</p>
        <p class="spinner-sub" id="spinner-sub">${sub}</p>
        <div class="spinner-dots">
          <span></span><span></span><span></span>
        </div>
      </div>`;
    document.body.appendChild(overlay);
  }
  $('#spinner-msg').textContent = msg;
  $('#spinner-sub').textContent = sub;
  overlay.classList.add('active');
}
function hideSpinner() {
  $('#spinner-overlay')?.classList.remove('active');
}

let _toastTimer = null; 
function showToast(msg, type = 'success') {
  let toast = $('#global-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'global-toast';
    toast.className = 'toast';
    document.body.appendChild(toast);
  }
  const icons = { success:'✅', error:'❌', info:'ℹ️', warn:'⚠️' };
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span class="toast-icon">${icons[type]||'💬'}</span><span>${msg}</span>`;
  toast.classList.add('show');
  clearTimeout(_toastTimer); 
  _toastTimer = setTimeout(() => toast.classList.remove('show'), 4200);
}

function initNav() {
  const nav = $('.nav');
  if (!nav) return;
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 30);
  }, { passive: true });
  const path = window.location.pathname.split('/').pop() || 'index.html';
  $$('.nav-link').forEach(link => {
    const href = link.getAttribute('href') || '';
    if (href === path || (path==='' && href==='index.html')) link.classList.add('active');
  });
}

function initReveal() {
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((entry, i) => {
      if (entry.isIntersecting) {
        setTimeout(() => entry.target.classList.add('visible'), i * 80);
        obs.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });
  $$('.reveal').forEach(el => obs.observe(el));
}

function initParticles(canvasId = 'particles-canvas') {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const resize = () => {
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
  };
  resize();
  window.addEventListener('resize', resize, { passive: true });

  const particles = Array.from({ length: 90 }, () => ({
    x:  Math.random() * canvas.width,
    y:  Math.random() * canvas.height,
    r:  Math.random() * 1.6 + 0.4,
    dx: (Math.random() - 0.5) * 0.28,
    dy: (Math.random() - 0.5) * 0.28,
    a:  Math.random() * 0.45 + 0.08,
  }));

  (function frame() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255,105,180,${p.a})`;
      ctx.fill();
      p.x += p.dx; p.y += p.dy;
      if (p.x < 0 || p.x > canvas.width)  p.dx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.dy *= -1;
    });
    requestAnimationFrame(frame);
  })();
}

async function apiCall(endpoint, method, payload) {
  const url  = `${API_BASE}${endpoint}`;
  const opts = { method };
  const controller = new AbortController();
  const timeoutId  = setTimeout(() => controller.abort(), 60_000);
  opts.signal = controller.signal;
  if (payload instanceof FormData) {
    opts.body = payload;
  } else if (payload) {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body    = JSON.stringify(payload);
  }
  try {
    const res  = await fetch(url, opts);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('Request timed out — server took too long to respond.');
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

async function checkHealth() {
  const badge = document.getElementById('server-status');
  try {
    const data = await apiCall('/health', 'GET');
    if (badge) {
      badge.textContent = data.demo_mode ? '⚡ Demo Mode' : '✅ AI Online';
      badge.className   = `badge ${data.demo_mode ? 'badge-gold' : 'badge-teal'}`;
    }
    return data;
  } catch {
    if (badge) { badge.textContent = '⚠️ Offline'; badge.className = 'badge badge-rose'; }
    return null;
  }
}

async function submitPrediction() {
  showSpinner('Running AI Analysis…', 'Fusing structured data + image models');

  try {
    const formData  = collectFormData();
    const imageFile = document.getElementById('ultrasound-image')?.files?.[0];
    const hasImage  = !!imageFile;
    
    if (imageFile) {
      const isLikelyGrayscale = await new Promise(resolve => {
        const reader = new FileReader();
        reader.onload = e => {
          const img = new Image();
          img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = 64; canvas.height = 64;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, 64, 64);
            const px = ctx.getImageData(0, 0, 64, 64).data;
            let totalDiff = 0;
            for (let i = 0; i < px.length; i += 4) {
              const r = px[i], g = px[i+1], b = px[i+2];
              totalDiff += Math.abs(r-g) + Math.abs(r-b) + Math.abs(g-b);
            }
            const avgDiff = totalDiff / (px.length / 4) / 3;
            resolve(avgDiff < 20); // grayscale threshold
          };
          img.src = e.target.result;
        };
        reader.readAsDataURL(imageFile);
     });

     if (!isLikelyGrayscale) {
        hideSpinner();
        showToast(
          '❌ Please upload a grayscale ultrasound image. Color photos are not accepted.',
          'error'
      );
    return; 
  }
}

    showSpinner('Step 1/3 – Structured Data…', 'Random Forest + XGBoost ensemble');
    const structuredResult = await apiCall('/predict_structured', 'POST', formData);

    let imageResult  = null;
    let uploadedB64  = null;

    if (hasImage) {
      showSpinner('Step 2/3 – Image Analysis…', 'EfficientNetB0 CNN processing');
      const fd = new FormData();
      fd.append('image', imageFile); 
      imageResult = await apiCall('/predict_image', 'POST', fd);

      uploadedB64 = await new Promise(resolve => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.readAsDataURL(imageFile);
      });
    }

    showSpinner('Step 3/3 – Fusing Results…', 'Late fusion + SHAP explainability');
    const fusionPayload = {
      structured: structuredResult,
      image:      imageResult,
      form_data:  formData,
    };
    const finalResult = await apiCall('/predict_final', 'POST', fusionPayload);

    finalResult.patient_name       = formData.patient_name;
    finalResult.doctor_name        = formData.doctor_name;
    finalResult.uploaded_image_b64 = uploadedB64;
    finalResult.uploaded_image_name= imageFile?.name || '';

    /* Store and redirect */
    STATE.lastResult = finalResult;
    sessionStorage.setItem('polycare_result', JSON.stringify(finalResult));
    hideSpinner();
    showToast('Analysis complete! Redirecting…', 'success');
    setTimeout(() => window.location.href = 'result.html', 800);

  } catch (err) {
    hideSpinner();
    console.error('Prediction error:', err);
    showToast(`Analysis failed: ${err.message}`, 'error');
  }
}

function nextStep() { /* overridden by predict.html */ }
function prevStep() { /* overridden by predict.html */ }

let _isDownloading = false;

async function downloadReport() {
  if (_isDownloading) {
    showToast('Report is already being generated…', 'warn');
    return;
  }

  const data = STATE.lastResult ||
               JSON.parse(sessionStorage.getItem('polycare_result') || 'null');
  if (!data) {
    showToast('No result data found. Run a diagnosis first.', 'warn');
    return;
  }

  _isDownloading = true;
  showSpinner('Building Clinical Report…', 'Preparing 2-page PDF');

  try {
    await loadScript('https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js');
    await loadScript('https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js');

    const isPcos   = data.prediction === 1;
    const conf     = data.confidence_pct || 0;
    const risk     = data.risk     || {};
    const fusion   = data.fusion   || {};
    const formData = data.form_data || {};
    const patient  = data.patient_name || 'Unknown';
    const doctor   = data.doctor_name  || 'N/A';

    const reportId = `PCR-${Date.now().toString().slice(-11)}`;
    const now      = new Date();
    const dateStr  = now.toLocaleDateString('en-IN', { day:'2-digit', month:'numeric', year:'numeric' });
    const genStr   = now.toLocaleDateString('en-IN', { day:'numeric', month:'long', year:'numeric' }) +
                     ' at ' + now.toLocaleTimeString('en-IN', { hour:'2-digit', minute:'2-digit', hour12:true });


    _setText('p1-report-id',      reportId);
    _setText('p1-gen-date',       genStr);
    _setText('p1-patient-name',   patient);
    _setText('p1-doctor-name',    doctor);
    _setText('p1-report-date',    dateStr);
    _setText('p1-report-id-meta', reportId);

    const banner = document.getElementById('p1-verdict-banner');
    if (banner) {
      banner.className = `p-verdict ${isPcos ? 'pcos-pos' : 'pcos-neg'}`;
    }
    const icon = document.getElementById('p1-verdict-icon');
    if (icon) icon.textContent = isPcos ? '⚠' : '✓';
    _setText('p1-verdict-label',
      isPcos ? 'PCOS POSITIVE — Indicators Detected' : 'PCOS NEGATIVE — No Indicators Detected');

    const verdictSubEl = document.getElementById('p1-verdict-sub');
    if (verdictSubEl) {
      verdictSubEl.textContent = isPcos
        ? 'AI analysis indicates polycystic ovarian syndrome indicators. Clinical correlation required.'
        : 'AI analysis does not detect significant PCOS indicators. Routine monitoring advised.';
    }
    _setText('p1-conf',     `${conf.toFixed(1)}%`);
    _setText('p1-risk',     risk.level || 'Unknown');
    _setText('p1-modality', fusion.has_image ? 'Multimodal (Clinical + Imaging)' : 'Structured Clinical Data');

    const clinRows = [
      ['Age',             formData.age           ? `${formData.age} years`           : '—'],
      ['Weight',          formData.weight         ? `${formData.weight} kg`           : '—'],
      ['Height',          formData.height         ? `${formData.height} cm`           : '—'],
      ['BMI',             formData.bmi            ? `${formData.bmi} kg/m²`          : '—'],
      ['Waist:Hip Ratio', formData['Waist:Hip Ratio'] ? `${parseFloat(formData['Waist:Hip Ratio']).toFixed(2)}` : '—'],
      ['Cycle Regularity',formData.cycle === 'I'  ? 'Irregular' : (formData.cycle === 'R' ? 'Regular' : '—')],
      ['Cycle Length',    formData.cycle_length   ? `${formData.cycle_length} days`  : '—'],
      ['Blood Pressure',  (formData.bp_systolic && formData.bp_diastolic)
                            ? `${formData.bp_systolic} / ${formData.bp_diastolic} mmHg` : '—'],
      ['Pulse Rate',      formData.pulse_rate     ? `${formData.pulse_rate} bpm`     : '—'],
    ];
    const clinEl = document.getElementById('p1-clinical-table');
    if (clinEl) {
      clinEl.innerHTML = clinRows.map(([k, v]) =>
        `<tr><td class="k">${k}</td><td class="v">${v}</td></tr>`
      ).join('');
    }
    const labDefs = [
      { n: 'LH (Luteinizing Hormone)',      v: formData.lh || formData.LH,        unit: 'mIU/mL', norm: '2–15',    lo: 2,    hi: 15   },
      { n: 'FSH (Follicle Stimulating H.)', v: formData.fsh || formData.FSH,       unit: 'mIU/mL', norm: '3–10',    lo: 3,    hi: 10   },
      { n: 'AMH (Anti-Müllerian H.)',       v: formData.amh || formData.AMH,       unit: 'ng/mL',  norm: '1–3.5',   lo: 1,    hi: 3.5  },
      { n: 'TSH (Thyroid Stimulating H.)',  v: formData.tsh || formData.TSH,       unit: 'mIU/L',  norm: '0.4–4.0', lo: 0.4,  hi: 4.0  },
      { n: 'Prolactin',                     v: formData.prolactin || formData.PRL, unit: 'ng/mL',  norm: '2–29',    lo: 2,    hi: 29   },
      { n: 'Blood Glucose (RBS)',           v: formData.rbs || formData.RBS,       unit: 'mg/dL',  norm: '70–140',  lo: 70,   hi: 140  },
      { n: 'Haemoglobin',                   v: formData.hb  || formData.hemoglobin,unit: 'g/dL',   norm: '12–16',   lo: 12,   hi: 16   },
      { n: 'Vitamin D3',                    v: formData.vitd3 || formData.vitD3,   unit: 'ng/mL',  norm: '—',       lo: null, hi: null },
    ].filter(r => r.v != null && r.v !== '');

    const labEl = document.getElementById('p1-lab-tbody');
    if (labEl) {
      labEl.innerHTML = labDefs.map(r => {
        const val = parseFloat(r.v);
        let status = 'Normal', cls = 's-ok';
        if (r.hi !== null && val > r.hi) { status = 'High'; cls = 's-high'; }
        else if (r.lo !== null && val < r.lo) { status = 'Low'; cls = 's-low'; }
        const ref = r.norm === '—' ? '—' : r.norm;
        return `<tr>
          <td>${r.n}</td>
          <td style="font-weight:700">${r.v} ${r.unit}</td>
          <td>${ref}</td>
          <td class="${cls}">${status}</td>
        </tr>`;
      }).join('');
    }



    _setText('p2-report-id',    reportId);
    _setText('p2-patient-name', patient);
    _setText('p2-sig-doctor',   `Dr. ${doctor}`);
    _setText('p2-sig-date',     dateStr);

    const usgImg    = document.getElementById('p2-usg-img');
    const usgWrap   = document.getElementById('p2-usg-wrap');
    const noImgEl   = document.getElementById('p2-no-image');
    const usgCaption = document.getElementById('p2-usg-caption');

    if (data.uploaded_image_b64) {
      if (usgImg)    { usgImg.src = data.uploaded_image_b64; }
      if (usgCaption){ usgCaption.textContent = `Uploaded: ${data.uploaded_image_name || 'ultrasound image'}`; }
      if (usgWrap)   usgWrap.style.display = 'flex';
      if (noImgEl)   noImgEl.style.display = 'none';
    } else {
      if (usgWrap)   usgWrap.style.display = 'none';
      if (noImgEl)   noImgEl.style.display = 'block';
    }

    const sympDefs = [
      { key: 'weight_gain',    label: 'Weight Gain'        },
      { key: 'hair_growth',    label: 'Excess Hair Growth'  },
      { key: 'skin_darkening', label: 'Skin Darkening'      },
      { key: 'pimples',        label: 'Acne / Pimples'      },
      { key: 'fast_food',      label: 'Fast Food Habit'     },
      { key: 'exercise',       label: 'Regular Exercise'    },
    ];
    const sympEl = document.getElementById('p2-symptoms');
    if (sympEl) {
      const chips = sympDefs
        .filter(s => formData[s.key] != null)
        .map(s => {
          const yes = parseInt(formData[s.key]) === 1;
          return `<span class="p-chip ${yes ? 'yes' : 'no'}">${yes ? '✓' : '✗'} ${s.label}</span>`;
        });
      sympEl.innerHTML = chips.length
        ? chips.join('')
        : '<span style="font-size:8px;color:#64748b;font-style:italic">No symptom data available.</span>';
    }

    const defaultRec = isPcos
      ? 'Multiple PCOS indicators detected. Please consult a gynaecologist urgently for hormonal evaluation, pelvic ultrasound confirmation, and metabolic assessment. Early intervention significantly improves long-term outcomes.'
      : 'No significant PCOS indicators detected. Maintain a healthy lifestyle and schedule routine follow-up with your gynaecologist. Continue monitoring menstrual regularity and hormonal health.';
    _setText('p2-recommendation', risk.recommendation || defaultRec);

    const usgDataEl = document.getElementById('p2-usg-table');
    if (usgDataEl) {
      const follL = formData.follicle_L ?? formData.follicle_l ?? '—';
      const follR = formData.follicle_R ?? formData.follicle_r ?? '—';
      const fsL   = formData.fsize_L   ?? formData.fsize_l   ?? '—';
      const fsR   = formData.fsize_R   ?? formData.fsize_r   ?? '—';
      const endT  = formData.endometrium ?? formData.endometrial_thickness ?? '—';
      usgDataEl.innerHTML = `
        <tr><td class="k">Follicle Count (R / L)</td><td class="v">${follR} / ${follL}</td></tr>
        <tr><td class="k">Avg. Follicle Size (R / L)</td><td class="v">${fsR} mm / ${fsL} mm</td></tr>
        <tr><td class="k">Endometrial Thickness</td><td class="v">${endT} mm</td></tr>`;
    }

    
    showSpinner('Rendering pages…', 'Converting HTML to PDF');

    const root = document.getElementById('polycare-pdf-root');
    root.style.display = 'block';
    await _sleep(400); 

    const { jsPDF: JsPDF } = window.jspdf;
    const pdf  = new JsPDF({ unit: 'mm', format: 'a4', orientation: 'portrait' });
    const A4W  = 210, A4H = 297;
    const pageIds = ['pdf-page-1', 'pdf-page-2'];

    for (let i = 0; i < pageIds.length; i++) {
      showSpinner(`Rendering page ${i + 1} of 2…`, 'Please wait');
      const pageEl = document.getElementById(pageIds[i]);

      const canvas = await html2canvas(pageEl, {
        scale:           2,
        useCORS:         true,
        allowTaint:      true,
        backgroundColor: '#ffffff',
        logging:         false,
        imageTimeout:    10000,
        windowWidth:     794,
        width:           794,
        height:          1123,
      });

      const imgData = canvas.toDataURL('image/jpeg', 0.93);
      if (i > 0) pdf.addPage();
      pdf.addImage(imgData, 'JPEG', 0, 0, A4W, A4H);
    }

    root.style.display = 'none';

    const safeName = patient.replace(/[^a-z0-9]/gi, '_').slice(0, 20);
    pdf.save(`POLYCARE_Report_${safeName}_${Date.now()}.pdf`);
    showToast('✅ Clinical report downloaded!', 'success');

  } catch (err) {
    const root = document.getElementById('polycare-pdf-root');
    if (root) root.style.display = 'none';
    showToast('PDF generation failed: ' + err.message, 'error');
    console.error('[POLYCARE PDF]', err);
  } finally {
    _isDownloading = false;
    hideSpinner();
  }
}
 
 
function _setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
 
function _setStyle(id, prop, val) {
  const el = document.getElementById(id);
  if (el) el.style[prop] = val;
}
 
function _sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
 

function loadScript(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
    const s = document.createElement('script');
    s.src = src; s.onload = resolve; s.onerror = reject;
    document.head.appendChild(s);
  });
}

function initHomeStats() {
  const statsSection = document.querySelector('.stats-section');
  if (!statsSection) return;
  const obs = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) {
      $$('[data-target]', statsSection).forEach(el => {
        const target = parseFloat(el.dataset.target);
        const dec    = parseInt(el.dataset.decimals) || 0;
        const dur    = 1800;
        const start  = performance.now();
        const fmt    = v => dec > 0 ? v.toFixed(dec) : Math.round(v);
        (function step(now) {
          const t = Math.min((now-start)/dur, 1);
          const e = 1 - Math.pow(1-t, 4);
          el.textContent = fmt(target * e);
          if (t < 1) requestAnimationFrame(step);
          else el.textContent = fmt(target);
        })(start);
      });
      obs.disconnect();
    }
  }, { threshold: 0.35 });
  obs.observe(statsSection);
}

function initChartDefaults() {
  if (!window.Chart) return;
  Chart.defaults.color       = '#7AAAA0';
  Chart.defaults.borderColor = 'rgba(0,212,170,0.07)';
  Chart.defaults.font.family = "'Figtree', sans-serif";
  Chart.defaults.font.size   = 12;
}

function initPredictPage() {
  if (!document.querySelector('.predict-page')) return;
}

function initResultPage() {
}

document.addEventListener('DOMContentLoaded', () => {
  initDarkMode();
  initNav();
  initReveal();
  initParticles();
  initChartDefaults();
  initHomeStats();
  initPredictPage();
  initResultPage();

  document.addEventListener('click', e => {
    if (e.target.closest('.dark-toggle')) toggleDarkMode();
  });
});

window.POLYCARE = {
  nextStep, prevStep,
  submitPrediction, downloadReport,
  toggleDarkMode, showToast, };

window._collectFormData_orig = window.collectFormData || function(){};

function collectFormData() {
  const g = id => {
    const el = document.getElementById(id);
    if (!el) return null;
    const v = el.value.trim();
    return v === '' ? null : isNaN(v) ? v : parseFloat(v);
  };

  const cycleRegVal = g('inp-cycle-reg');
  const cycleStr = cycleRegVal == 0 ? 'I' : 'R'; 

  const waist = g('inp-waist');
  const hip   = g('inp-hip');
  const whrDerived = (waist && hip && hip > 0) ? waist / hip : null;

  const prg = g('inp-prg');

  return {
    
    age:            g('inp-age') || 25,
    BMI:            g('inp-bmi'),
    weight:         g('inp-weight'),
    height:         g('inp-height'),
    cycle:          cycleStr,
    cycle_length:   g('inp-cycle'),
    LH:             g('inp-lh'),
    FSH:            g('inp-fsh'),
    AMH:            g('inp-amh'),

    
    TSH:            g('inp-tsh'),
    PRL:            g('inp-prolactin'),
    PRG:            prg,
    vitD3:          g('inp-vitd3'),
    RBS:            g('inp-glucose'),
    hemoglobin:     g('inp-hb'),

  
    follicle_L:     g('inp-follicle-l'),
    follicle_R:     g('inp-follicle-r'),
    fsize_L:        g('inp-fsize-l'),
    fsize_R:        g('inp-fsize-r'),
    endometrium:    g('inp-endometrium'),

    waist:          waist,
    hip:            hip,
    'Waist:Hip Ratio': whrDerived || g('inp-whr'),

    bp_systolic:    g('inp-sys-bp'),
    bp_diastolic:   g('inp-dia-bp'),
    pulse_rate:     g('inp-pulse'),

    weight_gain:    document.getElementById('sym-weight-gain')?.checked  ? 1 : (g('inp-skin-dark') != null ? 0 : null),
    hair_growth:    document.getElementById('sym-hair-growth')?.checked  ? 1 :
                    (document.getElementById('inp-hair-growth')?.value === '1' ? 1 : 0),
    skin_darkening: document.getElementById('sym-skin-dark')?.checked    ? 1 :
                    (document.getElementById('inp-skin-dark')?.value  === '1' ? 1 : 0),
    pimples:        document.getElementById('sym-pimples')?.checked       ? 1 : 0,
    fast_food:      document.getElementById('sym-fast-food')?.checked     ? 1 : 0,
    exercise:       document.getElementById('sym-exercise')?.checked      ? 1 : 0,

    patient_name:   document.getElementById('inp-patient-name')?.value.trim() || 'Unknown',
    doctor_name:    document.getElementById('inp-doctor-name')?.value.trim()  || 'N/A',

    form_data: {
      bmi:        g('inp-bmi'),
      lh:         g('inp-lh'),
      fsh:        g('inp-fsh'),
      amh:        g('inp-amh'),
      tsh:        g('inp-tsh'),
      prolactin:  g('inp-prolactin'),
      rbs:        g('inp-glucose'),
      hb:         g('inp-hb'),
      vitd3:      g('inp-vitd3'),
      bp_sys:     g('inp-sys-bp'),
      bp_dia:     g('inp-dia-bp'),
    }
  };
}

function _validateBeforeSubmit() {
  const lh  = parseFloat(document.getElementById('inp-lh')?.value)  || 0;
  const fsh = parseFloat(document.getElementById('inp-fsh')?.value) || 0;
  const amh = parseFloat(document.getElementById('inp-amh')?.value);
  const bmi = parseFloat(document.getElementById('inp-bmi')?.value);
  const wt  = parseFloat(document.getElementById('inp-weight')?.value);
  const ht  = parseFloat(document.getElementById('inp-height')?.value);
  const cycleReg = document.getElementById('inp-cycle-reg')?.value;

  const hasBmi    = (bmi > 0) || (wt > 0 && ht > 0);
  const hasCycle  = cycleReg !== '' && cycleReg != null;
  const hasLH     = lh > 0;
  const hasFSH    = fsh > 0;
  const hasAMH    = !isNaN(amh) && amh >= 0;

  const errors = [];
  if (!hasBmi)   errors.push('BMI (or Weight + Height)');
  if (!hasCycle) errors.push('Cycle Regularity');
  if (!hasLH)    errors.push('LH hormone level');
  if (!hasFSH)   errors.push('FSH hormone level');
  if (!hasAMH)   errors.push('AMH hormone level');

  return { valid: errors.length === 0, errors };
}

let _isSubmitting = false; 
async function submitPrediction() {

  if (_isSubmitting) {
    showToast('Analysis already in progress…', 'warn');
    return;
  }

  const { valid, errors } = _validateBeforeSubmit();
  if (!valid) {
    showToast(`Please fill required fields: ${errors.join(', ')}`, 'error');
    if (window.updateStepUI) {
      if (errors.some(e => e.includes('BMI') || e.includes('Cycle'))) {
        window.updateStepUI(1);
        setTimeout(() => {
          if (POLYCARE.showStep1Errors) POLYCARE.showStep1Errors();
        }, 100);
      } else {
        window.updateStepUI(2);
        setTimeout(() => {
          if (POLYCARE.showStep2Errors) POLYCARE.showStep2Errors();
        }, 100);
      }
    }
    return;
  }

  _isSubmitting = true; 
  showSpinner('Running AI Analysis…', 'Fusing structured data + image models');

  try {
    const formData  = collectFormData();
    const imageFile = document.getElementById('ultrasound-image')?.files?.[0];
    const hasImage  = !!imageFile;

    showSpinner('Step 1/3 – Structured Data…', 'Random Forest + XGBoost ensemble');
    const structuredResult = await apiCall('/predict_structured', 'POST', formData);

    let imageResult = null;
    let uploadedB64 = null;

    if (hasImage) {
      showSpinner('Step 2/3 – Image Analysis…', 'EfficientNetB0 CNN processing');
      const fd = new FormData();
      fd.append('image', imageFile);
      imageResult = await apiCall('/predict_image', 'POST', fd);

      uploadedB64 = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload  = e => resolve(e.target.result);
        reader.onerror = () => reject(new Error('Failed to read image file.'));
        reader.readAsDataURL(imageFile);
      });
    }

    showSpinner('Step 3/3 – Fusing Results…', 'Late fusion + SHAP explainability');
    const fd2 = new FormData();
    fd2.append('structured_data', JSON.stringify(formData));
    if (hasImage && imageFile) fd2.append('image', imageFile);

    const finalResult = await apiCall('/predict_final', 'POST', fd2);

    finalResult.patient_name        = formData.patient_name;
    finalResult.doctor_name         = formData.doctor_name;

    
    if (!finalResult.models || !finalResult.models.random_forest) {
      finalResult.models = structuredResult.models || finalResult.models || {};
    }

    
    if (!finalResult.shap || !finalResult.shap.top_features) {
      finalResult.shap = structuredResult.shap || finalResult.shap || null;
    }

    finalResult.uploaded_image_b64  = uploadedB64;
    finalResult.uploaded_image_name = imageFile?.name || '';

    finalResult.form_data           = formData;

    if (!finalResult.abnormal_features && structuredResult.abnormal_features) {
      finalResult.abnormal_features = structuredResult.abnormal_features;
    }

    console.log('[POLYCARE] downloadReport data:', finalResult);

    STATE.lastResult = finalResult;
    sessionStorage.setItem('polycare_result', JSON.stringify(finalResult));
    hideSpinner();
    showToast('Analysis complete! Redirecting…', 'success');
    setTimeout(() => window.location.href = 'result.html', 800);

  } catch (err) {
    hideSpinner();
    console.error('Prediction error:', err);
    showToast(`Analysis failed: ${err.message}`, 'error');
  } finally {
    _isSubmitting = false; 
  }
}
function renderDownloadSection() {
  const container = document.querySelector('.result-page .container');
  if (!container) return;
  if (document.querySelector('.download-section')) return;

  const section = document.createElement('div');
  section.className = 'download-section';
  section.innerHTML = `
    <div class="download-section-text">
      <h3>📋 Clinical Report Ready</h3>
      <p>Download a full PDF clinical summary — includes prediction verdict,
         confidence score, biomarker analysis, SHAP feature importance, and
         personalised recommendations.</p>
      <div class="download-section-checklist">
        <span>Prediction verdict &amp; confidence score</span>
        <span>Abnormal biomarker highlights</span>
        <span>SHAP feature importance chart</span>
        <span>Grad-CAM ultrasound heatmap (if image provided)</span>
        <span>Risk-stratified clinical recommendations</span>
      </div>
      <button class="download-cta-btn" onclick="POLYCARE.downloadReport()">
        📥 Download Clinical Report
      </button>
    </div>
    <div class="download-section-visual">📄</div>`;

  const actionBar = document.querySelector('.action-bar');
  if (actionBar) {
    container.insertBefore(section, actionBar);
  } else {
    container.appendChild(section);
  }
}

function renderFooter() {
  const footers = document.querySelectorAll('footer.footer');
  footers.forEach(footer => {
    const inner = footer.querySelector('.footer-inner');
    if (!inner) return;
    inner.innerHTML = `
      <div class="footer-brand">POLY<span style="color:var(--pink)">CARE</span></div>
      <div class="footer-links">
        <a href="index.html"   class="footer-link">Home</a>
        <a href="predict.html" class="footer-link">Run Diagnosis</a>
        <a href="about.html"   class="footer-link">About</a>
      </div>
      <div class="footer-copy">
        Developed by <strong>Sinchana H</strong> &amp; <strong>Tasmia Firdose A</strong>
      </div>
      <div class="footer-disclaimer">
        ⚕️ POLYCARE is for screening purposes only. Not a substitute for professional medical diagnosis.
        Always consult a qualified gynaecologist for clinical decisions.
      </div>`;
  });
}

document.addEventListener('DOMContentLoaded', () => {
  window.POLYCARE = Object.assign(window.POLYCARE || {}, {
    submitPrediction,
    downloadReport:    window.downloadReport || function(){},
    toggleDarkMode:    window.toggleDarkMode || function(){},
    showToast:         window.showToast      || function(){},
    nextStep:          window.POLYCARE?.nextStep  || window.nextStep  || function(){},
    prevStep:          window.POLYCARE?.prevStep  || window.prevStep  || function(){},
    clearForm:         window.POLYCARE?.clearForm || function(){},
    updateRatioHint:   window.POLYCARE?.updateRatioHint || function(){},
    checkStep1:        window.POLYCARE?.checkStep1 || function(){ return true; },
    checkStep2:        window.POLYCARE?.checkStep2 || function(){ return true; },
    showStep1Errors:   window.POLYCARE?.showStep1Errors || function(){},
    showStep2Errors:   window.POLYCARE?.showStep2Errors || function(){},
    autofillFromPdf:   window.POLYCARE?.autofillFromPdf || function(){},
    uploadPdfReport:   window.POLYCARE?.uploadPdfReport || function(){},
    togglePdfSection:  window.POLYCARE?.togglePdfSection || function(){},
    pdfDragOver:       window.POLYCARE?.pdfDragOver  || function(){},
    pdfDragLeave:      window.POLYCARE?.pdfDragLeave || function(){},
    pdfDrop:           window.POLYCARE?.pdfDrop      || function(){},
    autoBMI:           window.POLYCARE?.autoBMI      || function(){},
    renderExtractedValues: window.POLYCARE?.renderExtractedValues || function(){},
    showPdfStatus:         window.POLYCARE?.showPdfStatus || function(){},
  });

  renderFooter();

  if (document.querySelector('.result-page')) {
    renderDownloadSection();
  }
});