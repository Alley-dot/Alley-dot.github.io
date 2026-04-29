# 我的笔记博客

这是一个基于 Hexo 和 NexT 主题的个人笔记博客，适合长期写 Markdown 学习笔记。

## 本地启动

```bash
npm install
npm run dev
```

默认访问地址：`http://localhost:4000`

## 新建文章

```bash
npm run new -- "你的文章标题"
```

文章生成在 `source/_posts/`。

## 新建独立页面

```bash
npm run new:page -- about
```

## 构建静态文件

```bash
npm run build
```

构建产物位于 `public/`。

## 部署到 GitHub Pages

1. 创建仓库 `alley-dot.github.io`
2. 修改 `_config.yml` 中的 `url`
3. 修改 `_config.next.yml` 中的社交链接和邮箱
4. 推送到 GitHub
5. 在仓库 `Settings > Pages` 中将来源设置为 `GitHub Actions`

之后每次推送到 `main` 分支都会自动发布。
