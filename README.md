# Slides

一个仓库里放多份 [Slidev](https://sli.dev/) 演示文稿。每份文稿是一个文件夹，入口固定为 `slides.md`。

```text
slides/
  welcome/
    slides.md
    components/
    snippets/
    public/
  my-talk/
    slides.md
```

`slides/<名字>/` 就是这一份文稿的根目录。组件、代码片段和静态资源放在同一层，不要放到仓库根目录。文件夹名会成为访问路径，建议用英文短横线。

仓库根上原来的 `pages/` 已去掉。那是模板里用来把一份文稿拆成多个 Markdown 的示例，不是站点目录。某份文稿很长时，仍可以在它自己的目录里用 `src: ./other.md` 拆页。

## 本地预览

```bash
npm install
npm run dev -- welcome
```

## 构建

```bash
npm run build
```

产物在 `dist/`：`index.html` 是目录页，每份文稿在 `dist/<名字>/`。本地构建的基路径是 `/`。GitHub Actions 会把项目站的基路径设为 `/<仓库名>/`。

## 部署

推送到 `main` 后，`.github/workflows/deploy.yml` 会构建并发布到 GitHub Pages。仓库 Settings → Pages → Source 需要选 **GitHub Actions**。

发布后的地址：

- 目录：`https://<user>.github.io/<repo>/`
- 某一份：`https://<user>.github.io/<repo>/<名字>/`

线上使用 hash 路由，刷新某一页不会 404。
