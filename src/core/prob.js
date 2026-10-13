/*
 * StarAI · 概率统计内核 (Layer 1.2 概率与统计)
 * 纯 JS 零依赖。复用世界顶级结果：
 *  - 多元高斯密度用 Cholesky 稳定计算（避免 Σ^{-1} 直接求逆，数值稳定）
 *  - KL 散度、JS 散度、互信息，均满足非负性等不变量
 * 不变量：概率密度积分≈1（采样近似）；KL(p‖q)≥0；对称矩阵协方差重建。
 */
(function (global) {
  const StarAI = global.StarAI || (global.StarAI = {});
  const LA = (StarAI.LA) || require('./la.js');
  const Num = (StarAI.Num) || require('./num.js');
  const Prob = {};

  // 1D 高斯密度
  Prob.gaussianPdf = function (x, mu, sigma) {
    return Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2)) / (sigma * Math.sqrt(2 * Math.PI));
  };

  // 多元高斯对数密度（Cholesky 稳定）：x,k 维向量；mu 均值；Sigma 协方差
  // cholSolve 返回 z = Σ^{-1} d（完整解），故马氏距离 = dᵀ Σ^{-1} d = dᵀ z（而非 ‖z‖²）。
  Prob.mvgLogPdf = function (x, mu, Sigma) {
    const k = x.length;
    const L = LA.cholesky(Sigma);
    const d = x.map((v, i) => v - mu[i]);
    const z = LA.cholSolve(L, d);           // z = Σ^{-1} d
    let maha = 0; for (let i = 0; i < k; i++) maha += d[i] * z[i]; // = dᵀ Σ^{-1} d
    let logdet = 0; for (let i = 0; i < k; i++) logdet += Math.log(L[i][i]);
    return -0.5 * (k * Math.log(2 * Math.PI) + 2 * logdet + maha);
  };

  Prob.mvgPdf = function (x, mu, Sigma) { return Math.exp(Prob.mvgLogPdf(x, mu, Sigma)); };

  // Box–Muller 标准正态采样
  Prob.sampleStandardNormal = function () {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  // KL 散度（高斯分布解析）：KL(N0‖N1) = 0.5( tr(S1^{-1}S0) + (μ1-μ0)^T S1^{-1}(μ1-μ0) - k + ln|Σ1| - ln|Σ0| )
  Prob.klGaussian = function (mu0, S0, mu1, S1) {
    const k = mu0.length;
    const L1 = LA.cholesky(S1);
    const diff = mu1.map((v, i) => v - mu0[i]);
    const z = LA.cholSolve(L1, diff);            // z = Σ1^{-1} diff
    let maha = 0; for (let i = 0; i < k; i++) maha += diff[i] * z[i]; // = diffᵀ Σ1^{-1} diff
    // tr(S1^{-1} S0)：逐列解 S1 x_c = S0[:,c]，取对角
    let tr = 0;
    for (let c = 0; c < k; c++) {
      const b = S0.map(r => r[c]);
      const xc = LA.cholSolve(L1, b);
      tr += xc[c];
    }
    let logdet1 = 0, logdet0 = 0;
    for (let i = 0; i < k; i++) logdet1 += Math.log(L1[i][i]);
    const L0 = LA.cholesky(S0); for (let i = 0; i < k; i++) logdet0 += Math.log(L0[i][i]);
    return 0.5 * (maha + tr - k + 2 * (logdet1 - logdet0));
  };

  Prob.klCategorical = function (p, q) {
    let s = 0;
    for (let i = 0; i < p.length; i++) if (p[i] > 0) s += p[i] * Math.log(p[i] / q[i]);
    return s;
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = Prob;
  StarAI.Prob = Prob;
})(typeof window !== 'undefined' ? window : globalThis);
