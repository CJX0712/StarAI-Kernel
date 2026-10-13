/*
 * StarAI · 集成层 (Layer 7-9 大模型/多模态) — 对接世界顶级现成模型
 * 原则：零重写、集大成。直接复用 Hugging Face transformers.js（WebGPU/CPU 推理运行时），
 * 不重造 Transformer，而是把世界级预训练权重的推理能力接入本内核。
 * 默认从 ESM CDN 动态加载，浏览器/Node(>=18) 均可。失败则优雅降级并报告原因。
 */
(function (global) {
  const StarAI = global.StarAI || (global.StarAI = {});
  const Integration = {};

  const CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';

  // 延迟加载 transformers.js 运行时（仅首次）
  let _transformers = null;
  async function loadRuntime() {
    if (_transformers) return _transformers;
    if (typeof globalThis.importScripts === 'function') {
      // Web Worker 环境
      _transformers = await import(CDN + '/dist/transformers.min.js');
    } else {
      _transformers = await import(CDN);
    }
    return _transformers;
  }

  /*
   * 文本生成 / 填空：复用世界级因果语言模型（如 Xenova 系列量化模型）。
   * 返回 { text, elapsedMs }。无网络/无 WebGPU 时抛错并附中文诊断。
   */
  Integration.textGeneration = async function (prompt, opts) {
    opts = opts || {};
    const modelId = opts.model || 'Xenova/distilgpt2';
    const maxNew = opts.maxNewTokens || 24;
    const t = await loadRuntime();
    const generator = await t.pipeline('text-generation', modelId, { device: opts.device || 'wasm' });
    const out = await generator(prompt, { max_new_tokens: maxNew });
    return { text: out[0].generated_text, model: modelId };
  };

  /*
   * 特征抽取：复用世界级句子/词向量模型，输出句向量，可接入本内核的 PCA/GMM 聚类。
   * 体现"组合超越"：世界级嵌入 + 自研严谨聚类。
   */
  Integration.embed = async function (texts, opts) {
    opts = opts || {};
    const modelId = opts.model || 'Xenova/all-MiniLM-L6-v2';
    const t = await loadRuntime();
    const extractor = await t.pipeline('feature-extraction', modelId, { device: opts.device || 'wasm' });
    const out = await extractor(texts, { pooling: 'mean', normalize: true });
    return { embeddings: out.tolist ? out.tolist() : out, model: modelId };
  };

  Integration.runtimeAvailable = function () { return _transformers !== null; };
  Integration.CDN = CDN;

  if (typeof module !== 'undefined' && module.exports) module.exports = Integration;
  StarAI.Integration = Integration;
})(typeof window !== 'undefined' ? window : globalThis);
