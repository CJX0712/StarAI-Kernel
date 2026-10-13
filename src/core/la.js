/*
 * StarAI · 线性代数内核 (Layer 1.1 线性代数)
 * 纯 JS 零依赖。复用世界级算法：
 *  - Jacobi 旋转特征分解（经典数值线性代数，machine-epsilon 级精度）
 *  - 由 A^T A 的对称特征分解导出 SVD（Eckart–Young 最优截断的基础）
 *  - Cholesky 分解（正定阵的平方根，高斯推断基石）
 * 不变量：
 *  - 对称阵特征分解 A = V diag(λ) V^T，且 V 正交 (V^T V = I)
 *  - Cholesky A = L L^T 重建误差 ~ 1e-12
 *  - SVD A = U S V^T 重建误差 ~ 1e-10，奇异值非负且降序
 */
(function (global) {
  const StarAI = global.StarAI || (global.StarAI = {});
  const LA = {};

  LA.identity = function (n) {
    const I = [];
    for (let i = 0; i < n; i++) { I.push(new Array(n).fill(0)); I[i][i] = 1; }
    return I;
  };

  LA.transpose = function (A) {
    const m = A.length, n = A[0].length, T = [];
    for (let j = 0; j < n; j++) { const row = new Array(m); for (let i = 0; i < m; i++) row[i] = A[i][j]; T.push(row); }
    return T;
  };

  LA.matMul = function (A, B) {
    const m = A.length, n = B.length, p = B[0].length, C = [];
    for (let i = 0; i < m; i++) {
      const row = new Array(p).fill(0);
      for (let k = 0; k < n; k++) { const a = A[i][k]; if (a === 0) continue; for (let j = 0; j < p; j++) row[j] += a * B[k][j]; }
      C.push(row);
    }
    return C;
  };

  LA.matVec = function (A, x) {
    const m = A.length, n = A[0].length, y = new Array(m).fill(0);
    for (let i = 0; i < m; i++) { let s = 0; for (let j = 0; j < n; j++) s += A[i][j] * x[j]; y[i] = s; }
    return y;
  };

  LA.vecDot = function (a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  LA.norm = function (v) { return Math.sqrt(LA.vecDot(v, v)); };
  LA.axpy = function (a, x, y) { const o = new Array(x.length); for (let i = 0; i < x.length; i++) o[i] = a * x[i] + (y ? y[i] : 0); return o; };
  LA.matAdd = function (A, B) { return A.map((r, i) => r.map((v, j) => v + B[i][j])); };

  // Cholesky：返回下三角 L，使 A = L L^T。非正定则抛错。
  LA.cholesky = function (A) {
    const n = A.length, L = [];
    for (let i = 0; i < n; i++) L.push(new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) {
          if (s <= 0) throw new Error('矩阵非正定，Cholesky 失败');
          L[i][j] = Math.sqrt(s);
        } else {
          L[i][j] = s / L[j][j];
        }
      }
    }
    return L;
  };

  // 用 Cholesky 解 L L^T x = b（下三角前向 + 上三角后向）
  LA.cholSolve = function (L, b) {
    const n = L.length, y = new Array(n);
    for (let i = 0; i < n; i++) { let s = b[i]; for (let j = 0; j < i; j++) s -= L[i][j] * y[j]; y[i] = s / L[i][i]; }
    const x = new Array(n);
    for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let j = i + 1; j < n; j++) s -= L[j][i] * x[j]; x[i] = s / L[i][i]; }
    return x;
  };

  // Jacobi 旋转对称特征分解：A(n×n 对称) = V diag(λ) V^T，V 正交，λ 降序
  LA.symEig = function (A, maxSweeps, tol) {
    maxSweeps = maxSweeps || 100; tol = tol || 1e-14;
    const n = A.length;
    const a = A.map(r => r.slice());
    let V = LA.identity(n);
    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
      if (off < tol) break;
      for (let p = 0; p < n; p++) {
        for (let q = p + 1; q < n; q++) {
          const apq = a[p][q];
          if (Math.abs(apq) < 1e-300) continue;
          const app = a[p][p], aqq = a[q][q];
          const theta = (aqq - app) / (2 * apq);
          // 当 app==aqq 时 theta=0，sign(0)=0 会使 t=0（零旋转，无法消去非对角元）。
          // 此时需 ±45° 旋转，对应 t = sign(apq)。这是经典 Jacobi 的易错点。
          const t = (theta === 0) ? Math.sign(apq) : Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(1 + t * t);
          const s = t * c;
          const tau = s / (1 + c);
          for (let i = 0; i < n; i++) {
            if (i === p || i === q) continue;
            const aip = a[i][p], aiq = a[i][q];
            a[i][p] = aip - s * (aiq + tau * aip);
            a[i][q] = aiq + s * (aip - tau * aiq);
            a[p][i] = a[i][p]; // 保持对称
            a[q][i] = a[i][q];
          }
          a[p][p] = app - t * apq;
          a[q][q] = aqq + t * apq;
          a[p][q] = 0; a[q][p] = 0;
          for (let i = 0; i < n; i++) {
            const vip = V[i][p], viq = V[i][q];
            V[i][p] = vip - s * (viq + tau * vip);
            V[i][q] = viq + s * (vip - tau * viq);
          }
        }
      }
    }
    const eig = [];
    for (let i = 0; i < n; i++) eig.push({ val: a[i][i], vec: V.map(r => r[i]) });
    eig.sort((x, y) => y.val - x.val);
    return { values: eig.map(e => e.val), vectors: eig.map(e => e.vec) };
  };

  // 由一组正交向量补全为 n×n 正交基（Gram–Schmidt），用于 SVD 的 U 补全
  function orthonormalCompletion(seed, n) {
    const basis = seed.filter(Boolean).map(v => v.slice());
    for (let d = 0; d < n; d++) {
      const cand = new Array(n);
      for (let i = 0; i < n; i++) cand[i] = Math.sin((d + 1) * (i + 1) * 1.3) + 0.5 * Math.cos((d + 2) * i);
      for (let iter = 0; iter < 3; iter++) {
        for (const b of basis) { const proj = LA.vecDot(cand, b); for (let i = 0; i < n; i++) cand[i] -= proj * b[i]; }
      }
      const nv = LA.norm(cand);
      if (nv > 1e-8) { for (let i = 0; i < n; i++) cand[i] /= nv; basis.push(cand); }
      if (basis.length >= n) break;
    }
    while (basis.length < n) basis.push(new Array(n).fill(0));
    return basis;
  }

  // 完整 SVD：A(m×n) = U(m×m) S(m×n) V^T(n×n)，奇异值降序非负
  // U/S/V 均为行主序矩阵（行=数组元素），可直接参与 matMul。
  LA.svd = function (A) {
    const m = A.length, n = A[0].length;
    const At = LA.transpose(A);
    const AtA = LA.matMul(At, A); // n×n 对称
    const eig = LA.symEig(AtA);
    const values = eig.values, vectors = eig.vectors; // vectors[i] 为第 i 右奇异向量（长度 n）
    const s = values.map(v => Math.sqrt(Math.max(0, v)));
    // V 行主序：V[row=r][col=c] = vectors[c][r]
    const V = LA.transpose(vectors);
    // 计算 U 的列向量 u_i = A v_i / s_i
    const uCols = [];
    for (let i = 0; i < n; i++) {
      if (s[i] > 1e-12) uCols.push(LA.matVec(A, vectors[i]).map(v => v / s[i]));
      else uCols.push(null);
    }
    const seed = uCols.filter(Boolean);
    const Ucomp = orthonormalCompletion(seed, m); // 补全为 m 个正交列
    const cols = [];
    let ci = 0;
    for (let i = 0; i < n; i++) cols.push(uCols[i] || Ucomp[ci++]);
    for (let i = n; i < m; i++) cols.push(Ucomp[ci++]);
    // 转置为行主序 U：U[row][col] = cols[col][row]
    const U = [];
    for (let r = 0; r < m; r++) { const row = new Array(m); for (let c = 0; c < m; c++) row[c] = cols[c][r]; U.push(row); }
    // S 为 m×n 对角
    const S = [];
    for (let i = 0; i < m; i++) { const row = new Array(n).fill(0); if (i < n) row[i] = s[i]; S.push(row); }
    return { U, S, V, singularValues: s.slice(0, Math.min(m, n)) };
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = LA;
  StarAI.LA = LA;
})(typeof window !== 'undefined' ? window : globalThis);
