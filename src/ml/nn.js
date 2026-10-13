/*
 * StarAI · 神经网络 (Layer 4/5 监督学习) — 手写反向传播 + 梯度检验
 * 复用世界级自动微分思想（链式法则），并用数值梯度交叉验证：
 *  - 解析梯度 vs 中心差分数值梯度，最大相对误差 < 1e-6（梯度检验不变量）
 *  - 损失随训练单调下降（对足够小学习率）
 *  - XOR 可学习（非线性可分问题的标志性测试）
 */
(function (global) {
  const StarAI = global.StarAI || (global.StarAI = {});
  const Num = StarAI.Num;
  const NN = {};

  function randn() { // Box–Muller
    let u = 0, v = 0; while (u === 0) u = Math.random(); while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // 2 层 MLP：in -> hidden(tanh) -> out(sigmoid)
  NN.MLP = function (din, dhid, dout) {
    const W1 = [], b1 = [];
    for (let i = 0; i < dhid; i++) { W1.push(new Array(din).fill(0).map(() => randn() * 0.5)); b1.push(0); }
    const W2 = [], b2 = [];
    for (let i = 0; i < dout; i++) { W2.push(new Array(dhid).fill(0).map(() => randn() * 0.5)); b2.push(0); }

    function forward(x) {
      const z1 = new Array(dhid); for (let j = 0; j < dhid; j++) { let s = b1[j]; for (let i = 0; i < din; i++) s += W1[j][i] * x[i]; z1[j] = s; }
      const a1 = z1.map(Num.tanh);
      const z2 = new Array(dout); for (let k = 0; k < dout; k++) { let s = b2[k]; for (let j = 0; j < dhid; j++) s += W2[k][j] * a1[j]; z2[k] = s; }
      const a2 = z2.map(Num.sigmoid);
      return { a1, a2 };
    }

    function loss(X, Y) {
      let L = 0;
      for (let i = 0; i < X.length; i++) { const o = forward(X[i]).a2; for (let k = 0; k < dout; k++) { const y = Y[i][k]; L += -(y * Math.log(o[k] + 1e-12) + (1 - y) * Math.log(1 - o[k] + 1e-12)); } }
      return L / X.length;
    }

    // 单样本参数梯度
    function gradSample(x, y) {
      const f = forward(x);
      const gW1 = W1.map(r => new Array(r.length).fill(0)), gb1 = new Array(dhid).fill(0);
      const gW2 = W2.map(r => new Array(r.length).fill(0)), gb2 = new Array(dout).fill(0);
      const dz2 = new Array(dout);
      for (let k = 0; k < dout; k++) dz2[k] = (f.a2[k] - y[k]); // sigmoid + BCE
      for (let k = 0; k < dout; k++) { gb2[k] += dz2[k]; for (let j = 0; j < dhid; j++) gW2[k][j] += dz2[k] * f.a1[j]; }
      const da1 = new Array(dhid);
      for (let j = 0; j < dhid; j++) { let s = 0; for (let k = 0; k < dout; k++) s += W2[k][j] * dz2[k]; da1[j] = s; }
      for (let j = 0; j < dhid; j++) { const dz1 = da1[j] * (1 - f.a1[j] * f.a1[j]); gb1[j] += dz1; for (let i = 0; i < din; i++) gW1[j][i] += dz1 * x[i]; }
      return { gW1, gb1, gW2, gb2 };
    }

    function train(X, Y, opts) {
      opts = opts || {};
      const lr = opts.lr || 0.5, epochs = opts.epochs || 5000;
      const losses = [];
      for (let e = 0; e < epochs; e++) {
        const gW1 = W1.map(r => r.map(() => 0)), gb1 = b1.map(() => 0);
        const gW2 = W2.map(r => r.map(() => 0)), gb2 = b2.map(() => 0);
        for (let i = 0; i < X.length; i++) {
          const g = gradSample(X[i], Y[i]);
          for (let j = 0; j < dhid; j++) { for (let p = 0; p < din; p++) gW1[j][p] += g.gW1[j][p]; gb1[j] += g.gb1[j]; }
          for (let k = 0; k < dout; k++) { for (let j = 0; j < dhid; j++) gW2[k][j] += g.gW2[k][j]; gb2[k] += g.gb2[k]; }
        }
        for (let j = 0; j < dhid; j++) { for (let p = 0; p < din; p++) W1[j][p] -= lr * gW1[j][p] / X.length; b1[j] -= lr * gb1[j] / X.length; }
        for (let k = 0; k < dout; k++) { for (let j = 0; j < dhid; j++) W2[k][j] -= lr * gW2[k][j] / X.length; b2[k] -= lr * gb2[k] / X.length; }
        if (e % 500 === 0) losses.push(loss(X, Y));
      }
      losses.push(loss(X, Y));
      return losses;
    }

    function gradientCheck(x, y, eps) {
      eps = eps || 1e-5;
      const g = gradSample(x, y);
      const params = [
        { get: () => W1, set: (v) => { for (let j = 0; j < dhid; j++) for (let p = 0; p < din; p++) W1[j][p] = v[j][p]; }, ana: g.gW1, name: 'W1' },
        { get: () => b1, set: (v) => { for (let j = 0; j < dhid; j++) b1[j] = v[j]; }, ana: g.gb1, name: 'b1' },
        { get: () => W2, set: (v) => { for (let k = 0; k < dout; k++) for (let j = 0; j < dhid; j++) W2[k][j] = v[k][j]; }, ana: g.gW2, name: 'W2' },
        { get: () => b2, set: (v) => { for (let k = 0; k < dout; k++) b2[k] = v[k]; }, ana: g.gb2, name: 'b2' }
      ];
      let maxRel = 0;
      for (const pr of params) {
        const cur = pr.get(); const backup = cur.map(r => r.slice ? r.slice() : r);
        for (let i = 0; i < cur.length; i++) {
          const ci = cur[i];
          for (let j = 0; j < ci.length; j++) {
            ci[j] += eps; const lp = loss([x], [y]);
            ci[j] -= 2 * eps; const lm = loss([x], [y]);
            ci[j] += eps;
            const num = (lp - lm) / (2 * eps);
            const ana = pr.ana[i][j];
            const denom = Math.max(1e-8, Math.abs(num) + Math.abs(ana));
            const rel = Math.abs(num - ana) / denom;
            if (rel > maxRel) maxRel = rel;
          }
        }
        pr.set(backup);
      }
      return maxRel;
    }

    return { forward, loss, train, gradientCheck, gradSample, predict: (x) => forward(x).a2,
      weights: () => ({ W1, b1, W2, b2 }), setWeights: (w) => { for (let j = 0; j < dhid; j++) { for (let p = 0; p < din; p++) W1[j][p] = w.W1[j][p]; b1[j] = w.b1[j]; } for (let k = 0; k < dout; k++) { for (let j = 0; j < dhid; j++) W2[k][j] = w.W2[k][j]; b2[k] = w.b2[k]; } } };
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = NN;
  StarAI.NN = NN;
})(typeof window !== 'undefined' ? window : globalThis);
