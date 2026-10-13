/*
 * StarAI · PCA (Layer 4 降维) — 基于 SVD 的世界级实现
 * 不变量：
 *  - 样本协方差矩阵对称且半正定
 *  - 截断秩-k 重建误差等于 Eckart–Young 最优（与全 SVD 重建差 < 1e-9）
 *  - 各主成分方向正交，解释方差之和 = 总方差
 */
(function (global) {
  const StarAI = global.StarAI || (global.StarAI = {});
  const LA = StarAI.LA, Num = StarAI.Num;
  const PCA = {};

  PCA.fit = function (X, opts) {
    opts = opts || {};
    const n = X.length, d = X[0].length;
    // 中心化
    const mean = new Array(d).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += X[i][j];
    for (let j = 0; j < d; j++) mean[j] /= n;
    const Xc = X.map(r => r.map((v, j) => v - mean[j]));
    // SVD
    const svd = LA.svd(Xc);
    const V = svd.V; // d×d，列=主成分方向
    const s = svd.singularValues; // 长度 min(n,d)
    const components = LA.transpose(V); // 每行一个主成分 (d×d)
    const scores = LA.matMul(Xc, V); // n×d
    // 解释方差
    const totalVar = Xc.reduce((acc, r) => acc + r.reduce((a, v) => a + v * v, 0), 0) / (n - 1);
    const explained = s.map(v => (v * v) / (n - 1));
    const explainedRatio = explained.map(v => v / totalVar);
    return {
      mean, components, scores, singularValues: s,
      explainedVariance: explained, explainedVarianceRatio: explainedRatio,
      totalVariance: totalVar,
      transform: function (Y) {
        return Y.map(r => {
          const c = r.map((v, j) => v - mean[j]);
          return LA.matVec(V, c);
        });
      },
      reconstruct: function (k) {
        k = k || d;
        // scores = Xc·V = U·S；rank-k 重建 = (scores[:,:k]) · (V^T)[:k,:]
        // 其中 (V^T)[:k,:] = 前 k 个主成分（行） = components.slice(0,k)
        const Pk = components.slice(0, k); // k×d，行=主成分
        const recC = LA.matMul(scores.map(r => r.slice(0, k)), Pk);
        return recC.map((r, i) => r.map((v, j) => v + mean[j]));
      }
    };
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = PCA;
  StarAI.PCA = PCA;
})(typeof window !== 'undefined' ? window : globalThis);
