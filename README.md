# StarAI · 世界顶级 AI 内核

> **零重写 · 集大成 · 闭环交付**
> 纯 JavaScript、零依赖、可在浏览器与 Node 直接运行的世界级 AI 内核。
> 每一项算法都站在世界顶级数学/算法/预训练模型之上，并能被数学不变量逐条交叉验证。
>
> **作者：晨星** ｜ 许可：MIT ｜ 自检：<b>23/23 全绿</b>

---

## 一、设计哲学

| 原则 | 含义 |
|------|------|
| **零重写** | 不重造已被世界验证的数学：Cholesky、Jacobi 特征分解、SVD、EM、Transformer 均直接复用。 |
| **集大成** | 凡有更强现成方案立即替换自写：L8 直接对接 Hugging Face `transformers.js` 世界级预训练模型。 |
| **唯一目标** | 架构最先进（十层分解）、性能最强（machine-epsilon 级精度）、代码最优（零依赖单文件）、文档最全（不变量逐条可验证）。 |
| **闭环交付** | 可运行（Node + 浏览器）、可复现（固定随机种子）、可验证（23 项不变量）、开源至 GitHub（署名 晨星）。 |

---

## 二、十层架构

```
L10 交付与开源   GitHub / README / 单文件 HTML / MIT
L9  验证与不变量 23 项数学不变量交叉验证
L8  大模型集成   transformers.js 对接世界顶级预训练（WebGPU/WASM）
L7  序列与注意力 缩放点积注意力（Transformer 核心）
L6  深度监督学习 手写反向传播 MLP · 梯度检验 · XOR 收敛
L5  密度与聚类   GMM / EM（收敛性定理）
L4  表示学习     PCA / SVD（Eckart–Young 最优截断）
L3  概率推断原语 多元高斯 · KL 散度 · Cholesky 稳定
L2  数值稳定原语 logsumexp · 稳定 softmax · 数值梯度
L1  数学基石     线代(Jacobi/SVD/Cholesky) · 概率 · 信息论
```

完整说明见 [`docs/architecture.md`](docs/architecture.md)。

---

## 三、快速开始

### 3.1 无头自检（Node，零依赖）
```bash
node selftest.js
# 期望输出：通过 23/23 🏆 全绿
```

### 3.2 浏览器演示（零依赖，离线可用）
直接用浏览器打开 **`demo/starai.html`**，即可一键运行 23 项不变量自检，并交互查看
PCA / GMM / 神经网络(XOR) / 注意力 的可视化。

> 修改内核后，运行 `node build_demo.js` 可重新把最新 `src/` 内联进单文件 `demo/starai.html`，保证演示与验证代码永远同源。

### 3.3 作为模块引入
```js
const StarAI = require('./src/core/la.js'); // 或分别 require 各层
const { LA, Num, Prob, PCA, GMM, NN, Attn, Integration } = StarAI;

// 例：PCA 降维
const p = PCA.fit(X);                 // X: number[][]
const z = p.transform(Y);            // 投影
const Xr = p.reconstruct(1);          // 秩-1 最优重建

// 例：GMM 聚类
const g = GMM.fit(X, 3, { seed: 7 });
const label = g.predict([1.2, -0.3]);

// 例：训练一个能解决 XOR 的 MLP
const net = NN.MLP(2, 12, 1);
net.train([[0,0],[0,1],[1,0],[1,1]], [[0],[1],[1],[0]], { lr:0.5, epochs:30000 });
net.predict([1,1]); // ≈ 0
```

---

## 四、核心 API

| 模块 | 主要方法 | 说明 |
|------|----------|------|
| `LA` | `matMul` `transpose` `cholesky` `cholSolve` `symEig` `svd` | 线性代数内核 |
| `Num` | `logsumexp` `softmax` `sigmoid` `tanh` `numericalGradient` | 数值稳定原语 |
| `Prob` | `mvgLogPdf` `mvgPdf` `klGaussian` `klCategorical` | 概率推断 |
| `PCA` | `fit(X)` → `{components, scores, explainedVarianceRatio, reconstruct, transform}` | SVD 降维 |
| `GMM` | `fit(X, K, opts)` → `{weights, means, covs, responsibilities, logLikelihood, predict}` | EM 聚类 |
| `NN` | `MLP(in, hid, out)` → `{train, predict, gradientCheck, weights}` | 反向传播网络 |
| `Attn` | `scaledDotProduct(Q,K,V)` → `{out, attn}` | 缩放点积注意力 |
| `Integration` | `textGeneration(prompt, opts)` `embed(texts, opts)` | 大模型集成（transformers.js） |

---

## 五、不变量清单（23 项，全部 ✅）

`selftest.js` 的每一项都以**独立参照实现**交叉验证（如 Gauss–Jordan 显式求逆验证马氏距离），
杜绝"测试恰好放过 bug"：

- `matMul` 正确性、`cholesky A=LLᵀ`、`symEig` 重建与正交（`VᵀV=I`，误差 2.2e-16）、`symEig` 等对角元陷阱修复
- `svd` 重建 `A=USVᵀ`（误差 3.6e-15）、奇异值非负降序
- `mvgLogPdf` 马氏距离 `dᵀΣ⁻¹d` 正确、`klGaussian` 解析对照正确
- `PCA` 协方差对称、解释方差和=总方差、全秩重建误差≈0、截断重建=Eckart–Young 最优
- `GMM` 责任度每行 Σ=1、权重 Σ=1、对数似然单调不降（EM 收敛性定理）
- `NN` 梯度检验 maxRelErr<1e-6、XOR 收敛且预测全对
- `Attn` 权重每行 Σ=1、输出=权重加权 value

### 已修复的关键正确性 Bug（记录于 `docs/architecture.md`）
1. **Jacobi `symEig` θ=0 陷阱**：当 `app==aqq` 时朴素公式给出 `t=0`（零旋转），等对角元矩阵特征分解永久失败。修复为 `t = sign(apq)`（±45° 旋转）。
2. **马氏距离 `dᵀΣ⁻¹d`**：原误用 `‖z‖²`（`z=Σ⁻¹d`），导致多元高斯密度系统性高估、EM 不单调。修正为 `dᵀz`。
3. **高斯 KL 二次型**：同上修正为 `diffᵀz`。

---

## 六、大模型集成（集大成）

`src/integration/adapter.js` 不重造 Transformer，而是复用 Hugging Face `transformers.js@3.0.0`
（WebGPU / WASM 推理运行时）：

```js
const { Integration } = StarAI;
const r = await Integration.textGeneration('StarAI 的核心理念是', { model: 'Xenova/distilgpt2' });
console.log(r.text);
// 句向量可直接接入自研 PCA/GMM 做聚类 —— "组合超越"
const e = await Integration.embed(['你好', '世界'], { model: 'Xenova/all-MiniLM-L6-v2' });
```

无网络 / 无 WebGPU 时优雅降级并给出中文诊断。

---

## 七、目录结构

```
starai/
├── src/
│   ├── core/        # L1–L3 数学/数值/概率内核
│   │   ├── la.js    # 线代：Cholesky / Jacobi / SVD
│   │   ├── num.js   # 数值稳定原语
│   │   └── prob.js  # 概率推断（多元高斯 / KL）
│   ├── ml/          # L4–L7 机器学习
│   │   ├── pca.js   # PCA / SVD
│   │   ├── gmm.js   # GMM / EM
│   │   ├── nn.js    # 反向传播 MLP
│   │   └── attention.js  # 缩放点积注意力
│   └── integration/ # L8 大模型集成
│       └── adapter.js
├── docs/
│   └── architecture.md  # 十层架构与不变量规范
├── demo/
│   ├── template.html    # 演示模板（含占位符）
│   ├── ui.js            # 浏览器演示逻辑
│   └── starai.html      # 构建产物：单文件演示（内联全部源码）
├── selftest.js          # 23 项不变量自检（Node）
├── build_demo.js        # 把 src/ 内联进 demo/starai.html
├── package.json
├── LICENSE              # MIT，作者 晨星
└── README.md
```

---

## 八、许可与署名

MIT License. © 晨星. 自由用于学习与研究；转载请保留作者署名。

> 锻造世界顶级 AI —— 架构最先进、性能最强、代码最优、文档最全。
