---
title: 开始写第一篇笔记
date: 2026-04-29 00:00:00
updated: 2026-04-29 00:00:00
description: 这是你的第一篇示例笔记，用来熟悉这个博客的写作方式。
categories:
  - Blog
  - Getting Started
tags:
  - 笔记
  - Hexo
---
这个博客已经可以开始写 Markdown 笔记了。

## 本地预览

```bash
npm install
npm run dev
```

启动后打开 `http://localhost:4000` 就能看到页面效果。

## 新建一篇笔记

```bash
npm run new -- "强化学习 Chapter 1"
```

新文章会生成在 `source/_posts/` 目录下，你只需要继续写 Markdown 正文即可。

## 常用 Front Matter

```yaml
---
title: 强化学习 Chapter 1
date: 2026-04-29 21:00:00
updated: 2026-04-29 21:00:00
description: 这一章主要记录 MDP、策略和值函数的基础概念。
categories:
  - 强化学习
tags:
  - RL
  - 笔记
---
```

## 图片怎么插入

由于已经开启了 `post_asset_folder`，你后面可以用下面的方式管理图文笔记：

```bash
npm run new -- "我的新笔记"
```

然后把图片放到同名资源目录里，再在正文中引用：

```md
![实验结果](我的新笔记/curve.png)
```

## 部署到 GitHub Pages

项目里已经帮你准备好了 GitHub Pages 的工作流。你只需要：

1. 创建仓库 `你的 GitHub 用户名.github.io`
2. 把根目录 `_config.yml` 里的 `url` 改成你的站点地址
3. 推送代码到 GitHub
4. 在仓库的 Pages 设置里选择 `GitHub Actions`

之后每次 `git push`，站点都会自动更新。
