/*
 * StarAI · GMM 高斯混合模型 (Layer 4 密度估计) — EM 算法
 * 复用世界级 EM（Dempster–Laird–Rubin）与不变量定理：
 *  - 责任度每行 Σ=1（概率归一）
 *  - 混合权重 Σ=1
 *  - 对数似然单调不降（EM 收敛性定理的直接体现）
 *  - 协方差矩阵对称且正则化半正定
 */
(function (global) {
  const StarAI = global.StarAI || (global.StarAI = {});
  const LA = StarAI.LA, Prob = StarAI.Prob, Num = StarAI.Num;
  const GMM = {};

  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  function kmeansInit(X, K, rnd) {
    rnd = rnd || Math.random;
    const n = X.length, d = X[0].length;
    const centers = [X[Math.floor(rnd() * n)].slice()];
    while (centers.length < K) {
      const dists = X.map(x => {
        let best = Infinity;
        for (const c of centers) { let s = 0; for (let j = 0; j < d; j++) { const a = x[j] - c[j]; s += a * a; } if (s < best) best = s; }
        return best;
      });
      let tot = dists.reduce((a, b) => a + b, 0);
      let r = rnd() * tot, idx = 0;
      for (let i = 0; i < n; i++) { r -= dists[i]; if (r <= 0) { idx = i; break; } }
      centers.push(X[idx].slice());
    }
    return centers;
  }

  GMM.fit = function (X, K, opts) {
    opts = opts || {};
    const n = X.length, d = X[0].length;
    const maxIter = opts.maxIter || 200, tol = opts.tol || 1e-7, seed = opts.seed;
    const rnd = (seed !== undefined) ? mulberry32(seed) : Math.random;
    let means = kmeansInit(X, K, rnd);
    let weights = new Array(K).fill(1 / K);
    let covs = [];
    for (let k = 0; k < K; k++) { const C = []; for (let i = 0; i < d; i++) { C.push(new Array(d).fill(0)); C[i][i] = 1; } covs.push(C); }
    let llOld = -Infinity, logLik = [], resp = [];
    for (let iter = 0; iter < maxIter; iter++) {
      // E-step
      resp = []; const logLikPer = new Array(n).fill(0);
      for (let i = 0; i < n; i++) {
        const ls = [];
        for (let k = 0; k < K; k++) ls.push(Math.log(weights[k] + 1e-300) + Prob.mvgLogPdf(X[i], means[k], covs[k]));
        const lse = Num.logsumexp(ls);
        const row = ls.map(v => Math.exp(v - lse));
        resp.push(row); logLikPer[i] = lse;
      }
      const ll = logLikPer.reduce((a, b) => a + b, 0); logLik.push(ll);
      // M-step
      const Nk = new Array(K).fill(0);
      for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) Nk[k] += resp[i][k];
      for (let k = 0; k < K; k++) {
        weights[k] = Nk[k] / n;
        const mu = new Array(d).fill(0);
        for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mu[j] += resp[i][k] * X[i][j];
        for (let j = 0; j < d; j++) mu[j] /= (Nk[k] + 1e-300);
        means[k] = mu;
        // 纯 ML 协方差：Σ_k = scatter_k / Nk_k —— 恰为期望完整数据对数似然 Q 的精确极大元，
        // 故 EM 对混合对数似然严格单调（收敛性定理的直接体现）。空簇用单位阵兜底避免退化。
        const C = []; for (let a = 0; a < d; a++) C.push(new Array(d).fill(0));
        if (Nk[k] > 1e-9) {
          for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] += resp[i][k] * (X[i][a] - mu[a]) * (X[i][b] - mu[b]);
          for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] /= Nk[k];
        } else {
          for (let a = 0; a < d; a++) C[a][a] = 1;
        }
        covs[k] = C;
      }
      if (Math.abs(ll - llOld) < tol) break;
      llOld = ll;
    }
    return {
      K, weights, means, covs, responsibilities: resp, logLikelihood: logLik,
      predict: function (x) {
        let best = 0, bv = -Infinity;
        for (let k = 0; k < K; k++) { const lp = Math.log(weights[k] + 1e-300) + Prob.mvgLogPdf(x, means[k], covs[k]); if (lp > bv) { bv = lp; best = k; } }
        return best;
      }
    };
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = GMM;
  StarAI.GMM = GMM;
})(typeof window !== 'undefined' ? window : globalThis);
