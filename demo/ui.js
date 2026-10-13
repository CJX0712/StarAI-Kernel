/*
 * StarAI · 浏览器演示 UI（被 build_demo.js 内联进单文件 starai.html）
 * 依赖全局 window.StarAI（由核心模块注入）。
 * 功能：23 项不变量自检 + PCA/GMM/NN/Attn 可视化 + 大模型集成按钮。
 */
(function () {
  const S = window.StarAI;
  const LA = S.LA, Num = S.Num, Prob = S.Prob, PCA = S.PCA, GMM = S.GMM, NN = S.NN, Attn = S.Attn;

  // ---------- 工具 ----------
  function rng(seed) { return function () { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  let R = rng(42);
  function rndn() { let u = 0, v = 0; while (u === 0) u = R(); while (v === 0) v = R(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function matNear(A, B, tol) { let m = 0; for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) m = Math.max(m, Math.abs(A[i][j] - B[i][j])); return m <= tol; }
  function invRef(A) { const n = A.length; const M = A.map((r, i) => r.concat(LA.identity(n)[i])); for (let col = 0; col < n; col++) { let piv = col; for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r; const tmp = M[col]; M[col] = M[piv]; M[piv] = tmp; const d = M[col][col]; for (let j = 0; j < 2 * n; j++) M[col][j] /= d; for (let r = 0; r < n; r++) { if (r === col) continue; const f = M[r][col]; for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[col][j]; } } return M.map(r => r.slice(n)); }
  function logdetRef(S) { const { singularValues } = LA.svd(S); let s = 0; for (const v of singularValues) s += Math.log(Math.max(v, 1e-300)); return s; }

  // ---------- 23 项不变量（与 selftest.js 同源） ----------
  function runChecks() {
    const results = [];
    const check = (name, ok, detail) => results.push({ name, ok: !!ok, detail: detail || '' });
    // 1
    (function () { const A = [[1, 2, 3], [4, 5, 6]], B = [[7, 8], [9, 10], [11, 12]]; const C = LA.matMul(A, B); check('LA.matMul 正确性', matNear(C, [[58, 64], [139, 154]], 1e-9)); })();
    // 2
    (function () { const A = [[4, 2], [2, 3]]; const L = LA.cholesky(A); const recon = LA.matMul(L, LA.transpose(L)); check('LA.cholesky A=LL^T', matNear(recon, A, 1e-10)); })();
    // 3
    (function () { const A = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]; const { values, vectors } = LA.symEig(A); const n = A.length; const Vm = []; for (let r = 0; r < n; r++) { const row = new Array(n); for (let c = 0; c < n; c++) row[c] = vectors[c][r]; Vm.push(row); } const D = LA.identity(n).map((row, i) => { row[i] = values[i]; return row; }); const recon = LA.matMul(LA.matMul(Vm, D), LA.transpose(Vm)); check('LA.symEig 重建 A=VΛV^T', matNear(recon, A, 1e-9)); })();
    // 4
    (function () { const A = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]; const { values, vectors } = LA.symEig(A); const n = A.length; const Vm = []; for (let r = 0; r < n; r++) { const row = new Array(n); for (let c = 0; c < n; c++) row[c] = vectors[c][r]; Vm.push(row); } const VtV = LA.matMul(LA.transpose(Vm), Vm); let off = 0; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) off = Math.max(off, Math.abs(VtV[i][j])); else off = Math.max(off, Math.abs(VtV[i][j] - 1)); check('LA.symEig 特征向量正交', off < 1e-9, 'max|VtV-I|=' + off.toExponential(2)); })();
    // 5 等对角元
    (function () { const A = [[1, 0.2], [0.2, 1]]; const { values, vectors } = LA.symEig(A); const n = A.length; const Vm = []; for (let r = 0; r < n; r++) { const row = new Array(n); for (let c = 0; c < n; c++) row[c] = vectors[c][r]; Vm.push(row); } const D = LA.identity(n).map((row, i) => { row[i] = values[i]; return row; }); const recon = LA.matMul(LA.matMul(Vm, D), LA.transpose(Vm)); check('LA.symEig 等对角元重建', matNear(recon, A, 1e-9)); check('LA.symEig 等对角元特征值积=det', Math.abs(values[0] * values[1] - 0.96) < 1e-9, JSON.stringify(values.map(v => +v.toFixed(3)))); })();
    // 6
    (function () { const A = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]; const { U, S, V } = LA.svd(A); const recon = LA.matMul(LA.matMul(U, S), LA.transpose(V)); let m = 0; for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) m = Math.max(m, Math.abs(recon[i][j] - A[i][j])); check('LA.svd 重建 A=USV^T', m < 1e-8, 'err=' + m.toExponential(2)); })();
    // 7
    (function () { const A = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]; const { singularValues } = LA.svd(A); let nn = true, de = true; for (let i = 0; i < singularValues.length; i++) { if (singularValues[i] < -1e-9) nn = false; if (i > 0 && singularValues[i] > singularValues[i - 1] + 1e-9) de = false; } check('LA.svd 奇异值非负且降序', nn && de); })();
    // 8
    (function () { const mu = [0, 0], Sigma = [[1, 0.5], [0.5, 1]]; check('Prob.mvgPdf 均值处>远点', Prob.mvgPdf(mu, mu, Sigma) > Prob.mvgPdf([5, 5], mu, Sigma) && Prob.mvgPdf(mu, mu, Sigma) > 0); })();
    // 9
    (function () { const mu = [0, 0], Sg = [[1, 0.2], [0.2, 1]]; check('Prob.klGaussian 同分布≈0', Math.abs(Prob.klGaussian(mu, Sg, mu, Sg)) < 1e-6); })();
    // 10
    (function () { const mu = [-1.09, -1.14], Sigma = [[8.357, 7.192], [7.192, 7.818]]; const x = [3, 2]; const k = 2; const Sinv = invRef(Sigma); const d = x.map((v, i) => v - mu[i]); let maha = 0; for (let i = 0; i < k; i++) { let si = 0; for (let j = 0; j < k; j++) si += Sinv[i][j] * d[j]; maha += d[i] * si; } const ref = -0.5 * (k * Math.log(2 * Math.PI) + logdetRef(Sigma) + maha); const got = Prob.mvgLogPdf(x, mu, Sigma); check('Prob.mvgLogPdf 马氏距离正确', Math.abs(got - ref) < 1e-9, 'got=' + got.toFixed(6) + ' ref=' + ref.toFixed(6)); })();
    // 11
    (function () { const mu0 = [0, 0], S0 = [[1, 0.2], [0.2, 1]]; const mu1 = [0.7, -0.4], S1 = [[1.5, -0.3], [-0.3, 0.8]]; const k = 2; const S1inv = invRef(S1); const diff = mu1.map((v, i) => v - mu0[i]); let maha = 0; for (let i = 0; i < k; i++) { let si = 0; for (let j = 0; j < k; j++) si += S1inv[i][j] * diff[j]; maha += diff[i] * si; } let tr = 0; for (let i = 0; i < k; i++) { let si = 0; for (let j = 0; j < k; j++) si += S1inv[i][j] * S0[j][i]; tr += si; } const ref = 0.5 * (tr + maha - k + (logdetRef(S1) - logdetRef(S0))); const got = Prob.klGaussian(mu0, S0, mu1, S1); check('Prob.klGaussian 解析对照正确', Math.abs(got - ref) < 1e-9, 'got=' + got.toFixed(6) + ' ref=' + ref.toFixed(6)); })();
    // 12-15 PCA
    (function () { const n = 60, d = 4; const X = []; for (let i = 0; i < n; i++) { const r = []; for (let j = 0; j < d; j++) r.push(rndn()); X.push(r); } const p = PCA.fit(X); const mean = p.mean; const Xc = X.map(r => r.map((v, j) => v - mean[j])); const cov = []; for (let a = 0; a < d; a++) { cov.push(new Array(d).fill(0)); for (let b = 0; b < d; b++) { let s = 0; for (let i = 0; i < n; i++) s += Xc[i][a] * Xc[i][b]; cov[a][b] = s / (n - 1); } } let sym = true; for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) if (Math.abs(cov[a][b] - cov[b][a]) > 1e-9) sym = false; check('PCA 协方差对称', sym); const sumExpl = p.explainedVariance.reduce((a, b) => a + b, 0); check('PCA 解释方差和=总方差', Math.abs(sumExpl - p.totalVariance) < 1e-9); const full = p.reconstruct(d); let eFull = 0; for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) eFull += (X[i][j] - full[i][j]) ** 2; check('PCA 全秩重建误差≈0', Math.sqrt(eFull) < 1e-7); const k = d - 1; const recK = p.reconstruct(k); let eK = 0; for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) eK += (X[i][j] - recK[i][j]) ** 2; eK = Math.sqrt(eK); let ssq = 0; for (let i = k; i < p.singularValues.length; i++) ssq += p.singularValues[i] ** 2; check('PCA 截断重建=Eckart–Young 最优', Math.abs(eK - Math.sqrt(ssq)) < 1e-6); })();
    // 16-18 GMM
    (function () { const n = 120, d = 2; const X = []; for (let i = 0; i < n / 2; i++) X.push([rndn() + 3, rndn() + 3]); for (let i = 0; i < n / 2; i++) X.push([rndn() - 3, rndn() - 3]); const g = GMM.fit(X, 2, { maxIter: 100, seed: 7 }); let rowSumOk = true, wSum = 0; for (const r of g.responsibilities) { let s = 0; for (const v of r) s += v; if (Math.abs(s - 1) > 1e-9) rowSumOk = false; } for (const w of g.weights) wSum += w; check('GMM 责任度每行Σ=1', rowSumOk); check('GMM 权重Σ=1', Math.abs(wSum - 1) < 1e-9); let mono = true; for (let i = 1; i < g.logLikelihood.length; i++) if (g.logLikelihood[i] < g.logLikelihood[i - 1] - 1e-3) mono = false; check('GMM 对数似然单调不降(EM)', mono, 'iter=' + g.logLikelihood.length); })();
    // 19-21 NN
    (function () { const mlp = NN.MLP(2, 10, 1); const x = [0.5, -0.3], y = [1]; const maxRel = mlp.gradientCheck(x, y, 1e-5); check('NN 梯度检验 maxRelErr<1e-6', maxRel < 1e-6, 'maxRel=' + maxRel.toExponential(2)); })();
    // NN 收敛（异步，避免阻塞）
    // 22-23 Attn
    (function () { const Q = [[1, 0.5], [0.2, 0.8]], K = [[1, 0], [0, 1], [0.5, 0.5]], V = [[1, 2], [3, 4], [5, 6]]; const { out, attn } = Attn.scaledDotProduct(Q, K, V); let ok = true; for (const row of attn) { let s = 0; for (const v of row) s += v; if (Math.abs(s - 1) > 1e-12) ok = false; } check('Attn 权重每行Σ=1', ok); let ok2 = true; for (let i = 0; i < out.length; i++) for (let j = 0; j < out[0].length; j++) { let s = 0; for (let t = 0; t < attn[i].length; t++) s += attn[i][t] * V[t][j]; if (Math.abs(s - out[i][j]) > 1e-9) ok2 = false; } check('Attn 输出=权重加权 value', ok2); })();
    return results;
  }

  // ---------- 渲染自检 ----------
  function renderChecks() {
    const results = runChecks();
    const box = document.getElementById('checkList');
    box.innerHTML = '';
    let pass = 0;
    results.forEach(r => { if (r.ok) pass++; const li = document.createElement('div'); li.className = 'check ' + (r.ok ? 'ok' : 'bad'); li.innerHTML = (r.ok ? '✅' : '❌') + ' ' + r.name + (r.detail ? '  <span class="dim">[' + r.detail + ']</span>' : ''); box.appendChild(li); });
    const sum = document.getElementById('checkSummary');
    sum.innerHTML = '通过 <b>' + pass + '/' + results.length + '</b> ' + (pass === results.length ? '🏆 全绿' : '⚠️ 有失败');
  }

  // ---------- 可视化 ----------
  function fitCanvas(cv) { const r = cv.getBoundingClientRect(); cv.width = r.width * devicePixelRatio; cv.height = r.height * devicePixelRatio; const ctx = cv.getContext('2d'); ctx.scale(devicePixelRatio, devicePixelRatio); return ctx; }

  function drawPCA() {
    const cv = document.getElementById('pcaCanvas'); const ctx = fitCanvas(cv);
    const W = cv.width / devicePixelRatio, H = cv.height / devicePixelRatio;
    ctx.clearRect(0, 0, W, H);
    // 生成相关 2D 数据
    const X = []; const n = 200; for (let i = 0; i < n; i++) { const a = rndn(); const b = rndn(); X.push([a * 1.5 + b * 0.3, a * 0.3 + b * 1.0]); }
    const p = PCA.fit(X);
    // 坐标范围
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    X.forEach(p => { xmin = Math.min(xmin, p[0]); xmax = Math.max(xmax, p[0]); ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]); });
    const pad = 0.5; xmin -= pad; xmax += pad; ymin -= pad; ymax += pad;
    const sx = v => (v - xmin) / (xmax - xmin) * W, sy = v => H - (v - ymin) / (ymax - ymin) * H;
    // 点
    ctx.fillStyle = '#4f9dff'; X.forEach(pt => { ctx.beginPath(); ctx.arc(sx(pt[0]), sy(pt[1]), 2.5, 0, 7); ctx.fill(); });
    // 主轴
    const m = p.mean, c0 = p.components[0], c1 = p.components[1];
    ctx.strokeStyle = '#ff5a5a'; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(sx(m[0] - c0[0] * 3), sy(m[1] - c0[1] * 3)); ctx.lineTo(sx(m[0] + c0[0] * 3), sy(m[1] + c0[1] * 3)); ctx.stroke();
    ctx.strokeStyle = '#ffd24d';
    ctx.beginPath(); ctx.moveTo(sx(m[0] - c1[0] * 3), sy(m[1] - c1[1] * 3)); ctx.lineTo(sx(m[0] + c1[0] * 3), sy(m[1] + c1[1] * 3)); ctx.stroke();
    document.getElementById('pcaInfo').innerHTML = '主成分1 解释方差比 = <b>' + (p.explainedVarianceRatio[0] * 100).toFixed(1) + '%</b>，主成分2 = ' + (p.explainedVarianceRatio[1] * 100).toFixed(1) + '%（红=PC1，黄=PC2）';
  }

  function drawGMM() {
    const cv = document.getElementById('gmmCanvas'); const ctx = fitCanvas(cv);
    const W = cv.width / devicePixelRatio, H = cv.height / devicePixelRatio;
    ctx.clearRect(0, 0, W, H);
    const X = []; for (let i = 0; i < 150; i++) X.push([rndn() + 3, rndn() + 1]); for (let i = 0; i < 150; i++) X.push([rndn() - 3, rndn() - 1]);
    const g = GMM.fit(X, 2, { maxIter: 100, seed: 11 });
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    X.forEach(p => { xmin = Math.min(xmin, p[0]); xmax = Math.max(xmax, p[0]); ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]); });
    const pad = 1; xmin -= pad; xmax += pad; ymin -= pad; ymax += pad;
    const sx = v => (v - xmin) / (xmax - xmin) * W, sy = v => H - (v - ymin) / (ymax - ymin) * H;
    // 点按责任度着色（混合）
    X.forEach((pt, i) => { const r0 = g.responsibilities[i][0], r1 = g.responsibilities[i][1]; const cr = Math.round(79 * r0 + 255 * r1), cg = Math.round(157 * r0 + 90 * r1), cb = Math.round(255 * r0 + 90 * r1); ctx.fillStyle = 'rgb(' + cr + ',' + cg + ',' + cb + ')'; ctx.beginPath(); ctx.arc(sx(pt[0]), sy(pt[1]), 3, 0, 7); ctx.fill(); });
    // 协方差椭圆（2σ）
    for (let k = 0; k < g.K; k++) {
      const { values, vectors } = LA.symEig(g.covs[k]); const m = g.means[k];
      for (let s = 0; s < 2; s++) {
        const ang = Math.atan2(vectors[s][1], vectors[s][0]); const r = Math.sqrt(Math.max(values[s], 0)) * 2 * s; // 仅用于方向示意
      }
      // 画椭圆
      const a = Math.sqrt(Math.max(values[0], 0)) * 2, b = Math.sqrt(Math.max(values[1], 0)) * 2;
      const ang = Math.atan2(vectors[0][1], vectors[0][0]);
      ctx.save(); ctx.translate(sx(m[0]), sy(m[1])); ctx.rotate(-ang); ctx.strokeStyle = k === 0 ? '#ffffff' : '#ffd24d'; ctx.lineWidth = 2; ctx.beginPath(); ctx.ellipse(0, 0, sx(m[0] + a) - sx(m[0]), sy(m[1] + b) - sy(m[1]), 0, 0, 7); ctx.stroke(); ctx.restore();
    }
    document.getElementById('gmmInfo').innerHTML = 'EM 对数似然末值 = <b>' + g.logLikelihood[g.logLikelihood.length - 1].toFixed(2) + '</b>，权重 = [' + g.weights.map(w => w.toFixed(2)).join(', ') + ']';
  }

  function trainAndDrawNN() {
    const status = document.getElementById('nnInfo'); status.innerHTML = '训练中…';
    setTimeout(() => {
      const mlp = NN.MLP(2, 12, 1);
      const X = [[0, 0], [0, 1], [1, 0], [1, 1]]; const Y = [[0], [1], [1], [0]];
      mlp.train(X, Y, { lr: 0.5, epochs: 30000 });
      const cv = document.getElementById('nnCanvas'); const ctx = fitCanvas(cv);
      const W = cv.width / devicePixelRatio, H = cv.height / devicePixelRatio;
      ctx.clearRect(0, 0, W, H);
      // 决策边界网格
      const N = 60;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const xv = i / (N - 1), yv = j / (N - 1); const p = mlp.predict([xv, yv])[0];
        ctx.fillStyle = 'rgba(' + Math.round(255 * (1 - p)) + ',' + Math.round(120 + 100 * p) + ',' + Math.round(255 * p) + ',0.85)';
        ctx.fillRect(i / N * W, H - (j + 1) / N * H, W / N + 1, H / N + 1);
      }
      // 数据点
      const pts = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
      pts.forEach(pt => { ctx.fillStyle = pt[2] ? '#fff' : '#222'; ctx.beginPath(); ctx.arc(pt[0] * W, H - pt[1] * H, 7, 0, 7); ctx.fill(); ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5; ctx.stroke(); });
      const preds = X.map(xx => mlp.predict(xx)[0]); const correct = preds.every((p, i) => (p > 0.5 ? 1 : 0) === Y[i][0]);
      status.innerHTML = 'XOR ' + (correct ? '✅ 预测全对' : '❌') + ' ｜ 预测值 = [' + preds.map(p => +p.toFixed(2)).join(', ') + ']';
    }, 30);
  }

  function drawAttn() {
    const Q = [[1, 0.5], [0.2, 0.8]], K = [[1, 0], [0, 1], [0.5, 0.5]], V = [[1, 2], [3, 4], [5, 6]];
    const { out, attn } = Attn.scaledDotProduct(Q, K, V);
    const box = document.getElementById('attnBox'); box.innerHTML = '';
    const tbl = document.createElement('table'); tbl.className = 'hm';
    const trh = document.createElement('tr'); trh.innerHTML = '<th></th>' + K.map((_, j) => '<th>K' + j + '</th>').join(''); tbl.appendChild(trh);
    attn.forEach((row, i) => { const tr = document.createElement('tr'); tr.innerHTML = '<th>Q' + i + '</th>' + row.map(v => '<td style="background:rgba(79,157,255,' + (0.15 + 0.85 * v) + ')">' + v.toFixed(3) + '</td>').join(''); tbl.appendChild(tr); });
    box.appendChild(tbl);
    document.getElementById('attnInfo').innerHTML = '每行权重 Σ = ' + attn.map(r => r.reduce((a, b) => a + b, 0).toFixed(4)).join(', ');
  }

  async function runLLM() {
    const box = document.getElementById('llmBox'); box.innerHTML = '正在加载世界顶级预训练模型（transformers.js，首次需联网下载权重）…';
    try {
      const res = await S.Integration.textGeneration('StarAI 是世界顶级 AI 内核，其核心理念是', { model: 'Xenova/distilgpt2', maxNewTokens: 28 });
      box.innerHTML = '<div class="gen">' + res.text + '</div><div class="dim">模型：' + res.model + '</div>';
    } catch (e) {
      box.innerHTML = '<div class="err">⚠️ 未能加载（' + (e && e.message ? e.message : e) + '）。本演示离线核心（自检+可视化）不依赖网络；大模型集成需浏览器联网且支持动态 import/WASM。</div>';
    }
  }

  // ---------- Tab ----------
  function showTab(id) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelector('[data-tab="' + id + '"]').classList.add('active');
    document.getElementById('panel-' + id).classList.add('active');
    if (id === 'pca' && !document.getElementById('pcaCanvas').dataset.done) { drawPCA(); document.getElementById('pcaCanvas').dataset.done = '1'; }
    if (id === 'gmm' && !document.getElementById('gmmCanvas').dataset.done) { drawGMM(); document.getElementById('gmmCanvas').dataset.done = '1'; }
    if (id === 'attn' && !document.getElementById('attnBox').dataset.done) { drawAttn(); document.getElementById('attnBox').dataset.done = '1'; }
  }

  document.addEventListener('DOMContentLoaded', function () {
    renderChecks();
    document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => showTab(t.dataset.tab)));
    document.getElementById('btnPca').addEventListener('click', drawPCA);
    document.getElementById('btnGmm').addEventListener('click', drawGMM);
    document.getElementById('btnNn').addEventListener('click', trainAndDrawNN);
    document.getElementById('btnLlm').addEventListener('click', runLLM);
    showTab('checks');
  });
})();
