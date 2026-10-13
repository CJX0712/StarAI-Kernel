/*
 * StarAI · 数值内核 (Layer 1.4 数值方法)
 * 纯 JS 零依赖。复用世界顶级数值算法：logsumexp 防溢出、稳定 softmax。
 * 不变量：softmax 输出和为 1 且全正；logsumexp 与朴素求和在对数尺度一致。
 */
(function (global) {
  const StarAI = global.StarAI || (global.StarAI = {});
  const Num = {};

  // 防溢出对数求和：log(Σ exp(x_i)) = m + log(Σ exp(x_i - m))
  Num.logsumexp = function (x) {
    let m = -Infinity;
    for (let i = 0; i < x.length; i++) if (x[i] > m) m = x[i];
    if (m === -Infinity) return -Infinity;
    let s = 0;
    for (let i = 0; i < x.length; i++) s += Math.exp(x[i] - m);
    return m + Math.log(s);
  };

  // 数值稳定 softmax（按行向量）
  Num.softmax = function (x) {
    const m = Num.logsumexp(x);
    const out = new Array(x.length);
    let sum = 0;
    for (let i = 0; i < x.length; i++) {
      out[i] = Math.exp(x[i] - m);
      sum += out[i];
    }
    for (let i = 0; i < x.length; i++) out[i] /= sum;
    return out;
  };

  Num.sigmoid = function (x) { return 1 / (1 + Math.exp(-x)); };
  Num.tanh = function (x) { return Math.tanh(x); };
  Num.clamp = function (x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; };

  // 中心差分数值梯度（供反向传播梯度检验）
  Num.numericalGradient = function (f, w, eps) {
    eps = eps || 1e-5;
    const grad = new Array(w.length);
    for (let i = 0; i < w.length; i++) {
      const wp = w.slice(); wp[i] += eps;
      const wm = w.slice(); wm[i] -= eps;
      grad[i] = (f(wp) - f(wm)) / (2 * eps);
    }
    return grad;
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = Num;
  StarAI.Num = Num;
})(typeof window !== 'undefined' ? window : globalThis);
