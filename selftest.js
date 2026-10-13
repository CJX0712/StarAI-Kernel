/*
 * StarAI · 无头自检 runner (Node)
 * 运行全部数学不变量交叉验证，输出 全绿 / 失败。
 * 用法：node selftest.js
 */
'use strict';
const path = require('path');
// 加载全部模块（附加到 globalThis.StarAI）
require(path.join(__dirname, 'src/core/num.js'));
require(path.join(__dirname, 'src/core/la.js'));
require(path.join(__dirname, 'src/core/prob.js'));
require(path.join(__dirname, 'src/ml/pca.js'));
require(path.join(__dirname, 'src/ml/gmm.js'));
require(path.join(__dirname, 'src/ml/nn.js'));
require(path.join(__dirname, 'src/ml/attention.js'));

const S = globalThis.StarAI;
const LA = S.LA, Num = S.Num, Prob = S.Prob, PCA = S.PCA, GMM = S.GMM, NN = S.NN, Attn = S.Attn;

// 确定性 RNG（mulberry32）
function rng(seed) { return function () { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
let R = rng(42);
function rndn() { let u = 0, v = 0; while (u === 0) u = R(); while (v === 0) v = R(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function matNear(A, B, tol) { let m = 0; for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) m = Math.max(m, Math.abs(A[i][j] - B[i][j])); return m <= tol; }

// 独立参考：Gauss–Jordan 矩阵求逆（不依赖 StarAI 的 LA）
function invRef(A) {
  const n = A.length;
  const M = A.map((r, i) => r.concat(LA.identity(n)[i]));
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    const tmp = M[col]; M[col] = M[piv]; M[piv] = tmp;
    const d = M[col][col];
    for (let j = 0; j < 2 * n; j++) M[col][j] /= d;
    for (let r = 0; r < n; r++) { if (r === col) continue; const f = M[r][col]; for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[col][j]; }
  }
  return M.map(r => r.slice(n));
}
// 独立参考：行列式绝对值对数（由 SVD 奇异值）
function logdetRef(S) { const { singularValues } = LA.svd(S); let s = 0; for (const v of singularValues) s += Math.log(Math.max(v, 1e-300)); return s; }

const results = [];
function check(name, cond, detail) { results.push({ name, ok: !!cond, detail: detail || '' }); }

// 1. matMul
(function () {
  const A = [[1, 2, 3], [4, 5, 6]], B = [[7, 8], [9, 10], [11, 12]];
  const C = LA.matMul(A, B);
  const expect = [[58, 64], [139, 154]];
  check('LA.matMul 正确性', matNear(C, expect, 1e-9), JSON.stringify(C));
})();

// 2. Cholesky 重建
(function () {
  const A = [[4, 2], [2, 3]];
  const L = LA.cholesky(A);
  const recon = LA.matMul(L, LA.transpose(L));
  check('LA.cholesky A=LL^T', matNear(recon, A, 1e-10), '||A-LL^T||=' + (matNear(recon, A, 1e-10) ? 'ok' : 'bad'));
})();

// 3. 对称特征分解重建 + 正交
(function () {
  const A = [[2, 1, 0], [1, 3, 1], [0, 1, 2]];
  const { values, vectors } = LA.symEig(A);
  // 组装 V 矩阵（列=vector）
  const n = A.length; const Vm = [];
  for (let r = 0; r < n; r++) { const row = new Array(n); for (let c = 0; c < n; c++) row[c] = vectors[c][r]; Vm.push(row); }
  const D = LA.identity(n).map((row, i) => { row[i] = values[i]; return row; });
  const recon = LA.matMul(LA.matMul(Vm, D), LA.transpose(Vm));
  check('LA.symEig 重建 A=VΛV^T', matNear(recon, A, 1e-9), '');
  // 正交性 V^T V = I
  const VtV = LA.matMul(LA.transpose(Vm), Vm);
  let off = 0; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) off = Math.max(off, Math.abs(VtV[i][j])); else off = Math.max(off, Math.abs(VtV[i][j] - 1));
  check('LA.symEig 特征向量正交', off < 1e-9, 'max|VtV-I|=' + off.toExponential(2));
})();

// 3b. symEig 等对角元矩阵（经典 Jacobi θ=0 陷阱：需 ±45° 旋转）
(function () {
  const A = [[1, 0.2], [0.2, 1]];
  const { values, vectors } = LA.symEig(A);
  const n = A.length, Vm = [];
  for (let r = 0; r < n; r++) { const row = new Array(n); for (let c = 0; c < n; c++) row[c] = vectors[c][r]; Vm.push(row); }
  const D = LA.identity(n).map((row, i) => { row[i] = values[i]; return row; });
  const recon = LA.matMul(LA.matMul(Vm, D), LA.transpose(Vm));
  check('LA.symEig 等对角元重建 A=VΛV^T', matNear(recon, A, 1e-9), '');
  check('LA.symEig 等对角元特征值积=det', Math.abs(values[0] * values[1] - (1 * 1 - 0.04)) < 1e-9, JSON.stringify(values.map(v => +v.toFixed(3))));
})();

// 4. SVD 重建 + 奇异值非负降序
(function () {
  const A = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
  const { U, S, V, singularValues } = LA.svd(A);
  const recon = LA.matMul(LA.matMul(U, S), LA.transpose(V));
  check('LA.svd 重建 A=USV^T', matNear(recon, A, 1e-8), 'err=' + (function () { let m = 0; for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) m = Math.max(m, Math.abs(recon[i][j] - A[i][j])); return m.toExponential(2); })());
  let nonneg = true, desc = true;
  for (let i = 0; i < singularValues.length; i++) { if (singularValues[i] < -1e-9) nonneg = false; if (i > 0 && singularValues[i] > singularValues[i - 1] + 1e-9) desc = false; }
  check('LA.svd 奇异值非负且降序', nonneg && desc, JSON.stringify(singularValues.map(v => +v.toFixed(3))));
})();

// 5. 多元高斯：均值处密度 > 远点
(function () {
  const mu = [0, 0], Sigma = [[1, 0.5], [0.5, 1]];
  const atMean = Prob.mvgPdf(mu, mu, Sigma);
  const far = Prob.mvgPdf([5, 5], mu, Sigma);
  check('Prob.mvgPdf 均值处>远点', atMean > far && atMean > 0, '');
})();

// 6. KL 散度非负（同分布≈0）
(function () {
  const mu = [0, 0], S = [[1, 0.2], [0.2, 1]];
  const kl = Prob.klGaussian(mu, S, mu, S);
  check('Prob.klGaussian 同分布≈0', Math.abs(kl) < 1e-6, 'kl=' + kl.toExponential(2));
})();

// 6b. mvgLogPdf 与显式逆解（马氏距离 dᵀΣ⁻¹d）交叉验证 —— 捕获 zᵀz 误算 bug
(function () {
  const mu = [-1.09, -1.14], Sigma = [[8.357, 7.192], [7.192, 7.818]];
  const x = [3, 2];
  const k = x.length;
  const Sinv = invRef(Sigma);
  const d = x.map((v, i) => v - mu[i]);
  let maha = 0; for (let i = 0; i < k; i++) { let si = 0; for (let j = 0; j < k; j++) si += Sinv[i][j] * d[j]; maha += d[i] * si; }
  const ref = -0.5 * (k * Math.log(2 * Math.PI) + logdetRef(Sigma) + maha);
  const got = Prob.mvgLogPdf(x, mu, Sigma);
  check('Prob.mvgLogPdf 马氏距离正确(对照显式逆)', Math.abs(got - ref) < 1e-9, 'got=' + got.toFixed(6) + ' ref=' + ref.toFixed(6));
})();

// 6c. klGaussian 与解析参考交叉验证 —— 捕获二次型误算 bug
(function () {
  const mu0 = [0, 0], S0 = [[1, 0.2], [0.2, 1]];
  const mu1 = [0.7, -0.4], S1 = [[1.5, -0.3], [-0.3, 0.8]];
  const k = mu0.length;
  const S1inv = invRef(S1);
  const diff = mu1.map((v, i) => v - mu0[i]);
  let maha = 0; for (let i = 0; i < k; i++) { let si = 0; for (let j = 0; j < k; j++) si += S1inv[i][j] * diff[j]; maha += diff[i] * si; }
  let tr = 0; for (let i = 0; i < k; i++) { let si = 0; for (let j = 0; j < k; j++) si += S1inv[i][j] * S0[j][i]; tr += si; }
  const ref = 0.5 * (tr + maha - k + (logdetRef(S1) - logdetRef(S0)));
  const got = Prob.klGaussian(mu0, S0, mu1, S1);
  check('Prob.klGaussian 解析对照正确', Math.abs(got - ref) < 1e-9, 'got=' + got.toFixed(6) + ' ref=' + ref.toFixed(6));
})();

// 7. PCA：协方差对称、解释方差和=总方差、Eckart–Young 最优
(function () {
  const n = 60, d = 4; const X = [];
  for (let i = 0; i < n; i++) { const r = []; for (let j = 0; j < d; j++) r.push(rndn()); X.push(r); }
  const p = PCA.fit(X);
  // 协方差对称
  const mean = p.mean, Xc = X.map(r => r.map((v, j) => v - mean[j]));
  const cov = []; for (let a = 0; a < d; a++) { cov.push(new Array(d).fill(0)); for (let b = 0; b < d; b++) { let s = 0; for (let i = 0; i < n; i++) s += Xc[i][a] * Xc[i][b]; cov[a][b] = s / (n - 1); } }
  let sym = true; for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) if (Math.abs(cov[a][b] - cov[b][a]) > 1e-9) sym = false;
  check('PCA 协方差对称', sym, '');
  const sumExpl = p.explainedVariance.reduce((a, b) => a + b, 0);
  check('PCA 解释方差和=总方差', Math.abs(sumExpl - p.totalVariance) < 1e-9, '');
  // 全秩重建误差≈0
  const full = p.reconstruct(d);
  let eFull = 0; for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) eFull += (X[i][j] - full[i][j]) ** 2;
  check('PCA 全秩重建误差≈0', Math.sqrt(eFull) < 1e-7, 'e=' + Math.sqrt(eFull).toExponential(2));
  // 秩 d-1 重建误差 = Eckart-Young 最优（丢弃最小奇异值）
  const k = d - 1; const recK = p.reconstruct(k);
  let eK = 0; for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) eK += (X[i][j] - recK[i][j]) ** 2;
  eK = Math.sqrt(eK);
  const dropped = p.singularValues[k]; // 第 k 个（最小），因降序
  const opt = Math.sqrt(dropped * dropped * (n - 1)); // Xc 奇异值 s，重建误差=sqrt(Σ dropped^2)
  // 注意 singularValues 与 Xc 的奇异值一致，重建误差 Frobenius = sqrt(Σ_{i>k} s_i^2)
  let ssq = 0; for (let i = k; i < p.singularValues.length; i++) ssq += p.singularValues[i] ** 2;
  const opt2 = Math.sqrt(ssq);
  check('PCA 截断重建=Eckart–Young 最优', Math.abs(eK - opt2) < 1e-6, 'e=' + eK.toExponential(2) + ' opt=' + opt2.toExponential(2));
})();

// 8. GMM：责任度和=1、权重和=1、对数似然单调
(function () {
  const n = 120, d = 2; const X = [];
  for (let i = 0; i < n / 2; i++) X.push([rndn() + 3, rndn() + 3]);
  for (let i = 0; i < n / 2; i++) X.push([rndn() - 3, rndn() - 3]);
  const g = GMM.fit(X, 2, { maxIter: 100, seed: 7 });
  let rowSumOk = true, wSum = 0;
  for (const r of g.responsibilities) { let s = 0; for (const v of r) s += v; if (Math.abs(s - 1) > 1e-9) rowSumOk = false; }
  for (const w of g.weights) wSum += w;
  check('GMM 责任度每行Σ=1', rowSumOk, '');
  check('GMM 权重Σ=1', Math.abs(wSum - 1) < 1e-9, '');
  let mono = true; for (let i = 1; i < g.logLikelihood.length; i++) if (g.logLikelihood[i] < g.logLikelihood[i - 1] - 1e-3) mono = false;
  check('GMM 对数似然单调不降(EM)', mono, 'iters=' + g.logLikelihood.length + ' 末值=' + g.logLikelihood[g.logLikelihood.length - 1].toFixed(3));
})();

// 9. NN：梯度检验 + XOR 收敛
(function () {
  const mlp = NN.MLP(2, 10, 1);
  const x = [0.5, -0.3], y = [1];
  const maxRel = mlp.gradientCheck(x, y, 1e-5);
  check('NN 梯度检验 maxRelErr<1e-6', maxRel < 1e-6, 'maxRel=' + maxRel.toExponential(2));
  const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const Y = [[0], [1], [1], [0]];
  const losses = mlp.train(X, Y, { lr: 0.5, epochs: 40000 });
  const finalLoss = losses[losses.length - 1];
  const preds = X.map(xx => mlp.predict(xx)[0]);
  const correct = preds.every((p, i) => (p > 0.5 ? 1 : 0) === Y[i][0]);
  check('NN XOR 收敛(loss<0.1)', finalLoss < 0.1, 'loss=' + finalLoss.toFixed(4));
  check('NN XOR 预测全对', correct, JSON.stringify(preds.map(p => +p.toFixed(2))));
})();

// 10. Attention：权重每行Σ=1
(function () {
  const Q = [[1, 0.5], [0.2, 0.8]], K = [[1, 0], [0, 1], [0.5, 0.5]], V = [[1, 2], [3, 4], [5, 6]];
  const { out, attn } = Attn.scaledDotProduct(Q, K, V);
  let ok = true; for (const row of attn) { let s = 0; for (const v of row) s += v; if (Math.abs(s - 1) > 1e-12) ok = false; }
  check('Attn 权重每行Σ=1', ok, '');
})();

// 输出
let pass = 0;
console.log('\n=== StarAI 自检报告 ===');
for (const r of results) {
  const tag = r.ok ? '✅' : '❌';
  console.log(tag + ' ' + r.name + (r.detail ? '  [' + r.detail + ']' : ''));
  if (r.ok) pass++;
}
console.log('----------------------');
console.log('通过 ' + pass + '/' + results.length + (pass === results.length ? '  🏆 全绿' : '  ⚠️ 有失败'));
process.exit(pass === results.length ? 0 : 1);
