<a href="https://github.com/ChenZiHong-Gavin/repo-template">
<img height=350 alt="A Repo Template" src="https://capsule-render.vercel.app/api?type=waving&color=ffdd7a&height=300&section=header&text=A%20Repo%20Template&fontSize=70&fontColor=0f0000&animation=fadeIn&fontAlignY=38&desc=Replace%20with%20your%20project%20logo&descAlignY=60&descAlign=50"></img></a>

<p align="center">
  <b>Put here a brief description of your project.</b>
  <br>
  <br>
   <a href="https://github.com/ChenZiHong-Gavin/repo-template">
    <img title="Stars" src="https://img.shields.io/github/stars/ChenZiHong-Gavin/repo-template.svg?style=social&label=Star">
  </a>
  <a href="https://github.com/ChenZiHong-Gavin/repo-template/fork">
    <img title="Forks" src="https://img.shields.io/github/forks/ChenZiHong-Gavin/repo-template.svg?style=social&label=Fork">
  </a>
  <a href="https://github.com/ChenZiHong-Gavin/repo-template/issues">
    <img title="Issues" src="https://img.shields.io/github/issues/ChenZiHong-Gavin/repo-template.svg?style=social&label=Issues">
  </a>
  <a href="https://github.com/ChenZiHong-Gavin/repo-template/issues">
    <img title="Closed Issues" src="https://img.shields.io/github/issues-closed/ChenZiHong-Gavin/repo-template.svg?style=social&label=Closed%20Issues">
  </a>
  <a href="https://github.com/ChenZiHong-Gavin/repo-template/pulls">
    <img title="Pull Requests" src="https://img.shields.io/github/issues-pr/ChenZiHong-Gavin/repo-template.svg?style=social&label=Pull%20Requests">
  </a>
  <a href="https://github.com/ChenZiHong-Gavin/repo-template">
    <img title="License" src="https://img.shields.io/github/license/ChenZiHong-Gavin/repo-template.svg?style=social&label=License">
  </a>
</p>  

# Template Title

<details open>
<summary><b>📚 目录</b></summary>

- 📝 [基础内容](#基础内容)
- 📌 [代码质量与规范](#代码质量与规范)
- 👥 [协作与社区](#协作与社区)
- 🔧 [项目管理](#项目管理)
- 📝 [文档与示例](#文档与示例)
- 🍀 [致谢](#致谢)

</details>

## 基础内容
```
.
├── README.md                 # 项目门面：徽章、简介、快速开始、API示例
├── LICENSE                   # 开源协议（MIT/Apache 2.0等）
└── .gitignore                # 忽略构建产物、密钥、环境文件
```

### README.md
在Github中，README 是项目的默认文件，它会在项目的首页显示。它包含了项目的基本信息、快速开始指南、API示例等。README 应该简洁明了，能够帮助用户快速了解项目的功能和使用方法。

```
# 项目名称

[![版本](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/用户名/项目名/releases)
[![许可证](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一句话描述项目的核心功能。

## 📖 简介

详细介绍项目的背景、解决的问题、主要功能和适用场景。建议控制在3-5句话内。

## ✨ 特性

- 🚀 核心特性1：简要说明
- 💡 核心特性2：简要说明
- 🔒 核心特性3：简要说明
- 📦 核心特性4：简要说明

## 🛠️ 快速开始

### 安装

```bash
# 使用 npm
npm install 包名

# 或使用 yarn
yarn add 包名
```

### LICENSE
在项目中，通常会包含一个 LICENSE 文件，用来指定项目的开源许可证。开源许可证就是一份法律文件，它明确告诉别人：你可以用我的代码做什么，不能做什么，以及必须遵守什么条件。
#### LISCENSE 的两大阵营
- 宽松式许可证 (Permissive Licenses)
  - 对用户限制最少。几乎可以拿代码做任何事，只要保留署名。
  - 代表： MIT, Apache 2.0, BSD。
- 保护式许可证 (Copyleft Licenses)
  - 如果使用了代码并发布了修改版，那修改版也必须开源，并且使用相同的许可证。
  - 代表： GPL, LGPL, AGPL。

![](images/github%20license.png)

**如何在 GitHub 上选择？**
如果你不想研究法律条文，GitHub 提供了一个极好的官方工具：[choosealicense.com](https://choosealicense.com/)。它会问你几个简单的问题，根据你的回答，给你推荐一个合适的许可证。

### .gitignore
在项目中，通常会包含一个 .gitignore 文件，用来指定哪些文件不应该被 Git 版本控制。例如，一些构建产物、密钥、环境文件等。

```
# 忽略所有 .DS_Store 文件
.DS_Store

# 忽略 node_modules 目录
node_modules

# 忽略所有 .env 文件
.env
```

## 代码质量与规范
```
├── .github/
│   └── workflows/
│       ├── ci.yml          # 主CI流程：测试、构建、安全检查
│       └── release.yml     # 自动发版、打标签、生成Release Notes
└──  .pre-commit-config.yaml # 本地代码提交前检查
```

### .github/workflows
在项目中，通常会包含一个 .github/workflows 目录，用来存放 GitHub Actions 工作流文件。GitHub Actions 是 GitHub 提供的一种自动化工具，它可以在代码仓库中定义一些任务，例如测试、构建、部署等。

Github Actions 介绍：https://docs.github.com/en/actions

#### ci.yml
`ci.yml` 文件用来定义 GitHub Actions 的工作流。这个文件会在每次代码提交或拉取请求时触发，用来运行测试、构建、安全检查等任务。

下面是一个简单的 `ci.yml` 文件示例。
```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          pytest tests/ -v
      
      - name: Lint check
        run: |
          flake8 src/
```
#### release.yml
`release.yml` 文件用来定义 GitHub Actions 的工作流。这个文件会在每次代码发布时触发，用来自动发版、打标签、生成Release Notes 等任务。

下面是一个简单的 `release.yml` 文件示例。它会在每次推送 v* 格式标签时触发，自动发版、打标签、生成Release Notes 等任务。

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'  # 推送 v1.0.0 格式标签时触发

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write  # 必需：允许创建 Release
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # 获取完整提交历史以生成 Release Notes
      
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          generate_release_notes: true  # 自动生成变更日志
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### .pre-commit-config.yaml
`pre-commit-config.yaml` 文件用来定义本地代码提交前的检查。这个文件会在每次代码提交前运行，用来检查代码是否符合规范。

pre-commit 介绍：https://pre-commit.com/

下面是一个简单的 `pre-commit-config.yaml` 文件示例。
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
```

**如何安装和使用 pre-commit？**
1. 安装 pre-commit：`pip install pre-commit`
2. 在项目根目录下运行 `pre-commit install` 安装 pre-commit 钩子。
3. 每次代码提交前，pre-commit 会自动运行定义的检查。如果有问题，会提示你修复。


## 协作与社区
```
.github/
├── ISSUE_TEMPLATE/
│   ├── bug_report.yml
│   └── feature_request.yml
├── PULL_REQUEST_TEMPLATE.md
├── CONTRIBUTING.md
└── CODE_OF_CONDUCT.md
```

### .github/ISSUE_TEMPLATE
在项目中，通常会包含一个 .github/ISSUE_TEMPLATE 目录，用来存放 issue 模板文件。issue 模板文件用来定义 issue 的格式，例如 bug 报告、功能请求等。

下面是一个简单的 `bug_report.yml` 文件示例。
```yaml
name: Bug Report
description: 报告一个Bug
title: "[Bug]: "
labels: ["bug"]
body:
  - type: markdown
    attributes:
      value: |
        感谢报告Bug！请填写以下信息帮助我们修复问题。
  
  - type: input
    id: version
    attributes:
      label: 版本信息
      description: 请提供项目的版本号（如 v1.0.0）
      placeholder: v1.0.0
    validations:
      required: true
  
  - type: dropdown
    id: python-version
    attributes:
      label: Python版本
      options:
        - "3.8"
        - "3.9"
        - "3.10"
        - "3.11"
        - "3.12"
    validations:
      required: true
  
  - type: textarea
    id: description
    attributes:
      label: 问题描述
      description: 清晰简洁地描述Bug
    validations:
      required: true
  
  - type: textarea
    id: reproduce
    attributes:
      label: 复现步骤
      description: 如何复现这个问题？
      placeholder: |
        1. 执行 '...'
        2. 输入 '....'
        3. 看到错误 '....'
    validations:
      required: true
  
  - type: textarea
    id: expected
    attributes:
      label: 期望的行为
      description: 描述你期望的结果
  
  - type: textarea
    id: logs
    attributes:
      label: 相关日志
      description: 请粘贴相关日志（使用```包裹）
      render: shell
  
  - type: checkboxes
    id: terms
    attributes:
      label: 确认事项
      options:
        - label: 我已经搜索了现有的Issues
          required: true
        - label: 这是一个Bug报告，不是功能请求
          required: true
```

下面是一个简单的 `feature_request.yml` 文件示例。
```yaml
name: Feature Request
description: 建议一个新功能
title: "[Feature]: "
labels: ["enhancement"]
body:
  - type: markdown
    attributes:
      value: |
        感谢你的建议！请描述你想要的功能。
  
  - type: textarea
    id: problem
    attributes:
      label: 问题背景
      description: 这个功能解决什么问题？
    validations:
      required: true
  
  - type: textarea
    id: solution
    attributes:
      label: 建议的方案
      description: 你希望如何实现？
    validations:
      required: true
  
  - type: dropdown
    id: priority
    attributes:
      label: 优先级
      options:
        - 低
        - 中
        - 高
        - 紧急
    validations:
      required: true
  
  - type: checkboxes
    id: terms
    attributes:
      label: 确认事项
      options:
        - label: 我已经搜索了现有的Issues和PRs
          required: true
        - label: 我愿意实现这个功能
          required: false
```

### PULL_REQUEST_TEMPLATE.md
PULL_REQUEST_TEMPLATE.md 文件用来定义 pull request 的格式，描述 pull request 的内容，例如变更的功能、测试用例等。

```
## 描述

简要描述这个PR做了什么。

## 类型

- [ ] Bug修复
- [ ] 新功能
- [ ] 文档更新
- [ ] 代码重构
- [ ] 性能优化
- [ ] 测试增加

## 检查清单

- [ ] 代码遵循项目规范（black, ruff, mypy）
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 通过了CI检查
- [ ] 遵循了Conventional Commits规范

## 相关Issue

Closes #123
Related to #45

##  breaking changes

如果有破坏性变更，请在这里说明。

## 其他信息

任何其他需要reviewer知道的信息。
```

### CONTRIBUTING.md

CONTRIBUTING.md 文件用来描述如何贡献代码，例如提交 pull request、报告 bug 等。

````markdown
## 贡献流程
1. Fork 仓库
2. 创建分支：git checkout -b feat/your-feature-name
3. 提交更改：git commit -m "feat: add new feature"
4. 推送到 Fork：git push origin feat/your-feature-name
5. 创建 Pull Request 到主仓库的 main 分支

## Commit Message 规范
我们使用 Conventional Commits：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

类型说明：
- feat/ - 新功能
- fix/ - Bug 修复
- docs/ - 文档更新
- refactor/ - 代码重构
- test/ - 测试相关
- chore/ - 杂项（依赖更新等）

示例：
```
feat(auth): add JWT token validation

- Implement token verification
- Add token refresh endpoint

Closes #123
```

## Code Review 流程
1. 所有 PR 必须至少 1 个 approve 才能合并
2. CI 必须通过
3. 解决所有评论后才能合并
4. 使用 "Squash and merge" 合并

## 发布流程
只有 maintainer 可以发布新版本：
```
make release VERSION=1.0.0
```

````

### CODE_OF_CONDUCT.md

CODE_OF_CONDUCT.md 文件用来描述项目的行为规范，例如如何处理不遵守规范的行为等。

```
```markdown
# 行为准则

## 我们的承诺

我们致力于为所有人提供友好、安全和 welcoming 的环境。

## 我们的标准

**正面行为**：
- 使用 welcoming 和 inclusive 的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事

**不可接受的行为**：
- 使用 sexualized 语言或图像
- 人身攻击、侮辱性评论
- 公开或私下的骚扰
- 未经同意发布他人私人信息

## 执行

违反行为准则的维护者可能会被撤销权限。非维护者可能会被禁止参与。

## 报告

通过 issue 或邮件 your.email@example.com 报告不当行为。

## 归属

本行为准则改编自 [Contributor Covenant](https://www.contributor-covenant.org)。
```

## 项目管理
```
├── CHANGELOG.md            # 版本历史
├── ROADMAP.md              # 路线图
└── .github/
    └── labels.yml          # 标准化标签
```


### CHANGELOG.md
CHANGELOG.md 文件用来记录项目的版本历史，包括新增功能、修复的 bug、 Breaking Changes 等，由 `release.yml` 自动生成，无需手动维护。

```
# Changelog

所有重要变更都会记录在这里。

格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/spec/v2.0.0.html)。

## [Unreleased]

### Added
- 新功能...

### Changed
- 变更...

### Fixed
- Bug修复...

## [v1.0.0] - 2024-01-01

### Added
- 初始版本
- 核心功能实现
- CI/CD 配置

[v1.0.0]: https://github.com/你的用户名/项目名/releases/tag/v1.0.0
```


### ROADMAP.md
ROADMAP.md 文件用来记录项目的路线图，包括未来的功能、改进、维护等。

```
# 路线图

## 当前版本: v0.1.0

### 进行中 (In Progress)
- [ ] 功能A：描述...
- [ ] 功能B：描述...

### 计划中的功能 (Planned)
- [ ] v0.2.0 - 添加XX支持
- [ ] v0.3.0 - 优化XX性能
- [ ] v1.0.0 - 稳定版发布

### 未来想法 (Future Ideas)
- [ ] 支持XX平台
- [ ] 集成XX服务

## 如何提议新功能

请使用 [GitHub Issues](https://github.com/你的用户名/项目名/issues) 并打上 `enhancement` 标签。

## 版本规划

| 版本 | 预计发布 | 主要特性 |
|------|----------|----------|
| v0.2.0 | 2024-Q2 | XX, XX |
| v0.3.0 | 2024-Q3 | XX, XX |
| v1.0.0 | 2024-Q4 | 稳定API |
```

### .github/labels.yml
对于 github 上的 issue，我们进行标准化标签管理：

```
- name: "bug"
  color: "d73a4a"
  description: "Bug或错误"

- name: "enhancement"
  color: "a2eeef"
  description: "新功能或改进"

- name: "documentation"
  color: "0075ca"
  description: "文档相关"

- name: "good first issue"
  color: "7057ff"
  description: "适合新手的Issue"

- name: "help wanted"
  color: "008672"
  description: "需要帮助"

- name: "priority:high"
  color: "b60205"
  description: "高优先级"

- name: "priority:medium"
  color: "fbca04"
  description: "中优先级"

- name: "priority:low"
  color: "0e8a16"
  description: "低优先级"

- name: "status:blocked"
  color: "000000"
  description: "被阻塞"

- name: "status:in progress"
  color: "cccccc"
  description: "进行中"

- name: "type:feature"
  color: "5319e7"
  description: "功能类型"

- name: "type:bug"
  color: "b60205"
  description: "Bug类型"

- name: "type:refactor"
  color: "0052cc"
  description: "重构类型"
```

可以使用 github actions 自动管理 labels，具体配置可以参考 [actions/labeler](https://github.com/actions/labeler)。

`.github/workflows/labeler.yml`文件：
```
name: Labeler
on:
  pull_request:
    types: [opened]

jobs:
  label:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/labeler@v4
        with:
          repo-token: "${{ secrets.GITHUB_TOKEN }}"
          configuration-path: .github/labels.yml
```

## 文档与示例
```
.
├── docs/
│   ├── architecture.md     # 架构设计
│   ├── deployment.md       # 部署指南
│   ├── api.md              # API文档
│   └── contributing.md     # 贡献指南
├── examples/               # 可运行示例
│   ├── basic_usage.py
│   └── advanced_config.py
└── CONTRIBUTING.md         # 根目录贡献指南
```

根据项目的实际情况进行调整。


## 致谢
1. [MarketingPipeline/Awesome-Repo-Template](https://github.com/MarketingPipeline/Awesome-Repo-Template)
