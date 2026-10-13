/*
 * StarAI · 演示构建脚本
 * 将真实核心模块（src/）与浏览器 UI（demo/ui.js）内联进单文件 demo/starai.html。
 * 这样演示永远与经 23 项不变量验证的内核代码同源，杜绝复制漂移。
 * 用法：node build_demo.js
 */
'use strict';
const fs = require('fs');
const path = require('path');
const root = __dirname;

const mods = [
  'src/core/la.js',
  'src/core/num.js',
  'src/core/prob.js',
  'src/ml/pca.js',
  'src/ml/gmm.js',
  'src/ml/nn.js',
  'src/ml/attention.js',
  'src/integration/adapter.js'
];

const coreJs = mods.map(m => fs.readFileSync(path.join(root, m), 'utf8')).join('\n\n');
const uiJs = fs.readFileSync(path.join(root, 'demo/ui.js'), 'utf8');
const tpl = fs.readFileSync(path.join(root, 'demo/template.html'), 'utf8');

if (tpl.indexOf('/*__CORE__*/') < 0 || tpl.indexOf('/*__UI__*/') < 0) {
  throw new Error('模板缺少占位符 /*__CORE__*/ 或 /*__UI__*/');
}

const html = tpl.replace('/*__CORE__*/', coreJs).replace('/*__UI__*/', uiJs);
fs.writeFileSync(path.join(root, 'demo/starai.html'), html);
console.log('已生成 demo/starai.html （核心代码 ' + coreJs.length + ' 字节，UI ' + uiJs.length + ' 字节）');
