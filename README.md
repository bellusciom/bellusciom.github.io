# Personal project website

A fully static site (plain HTML + CSS, no build step) designed for GitHub Pages.

## Layout

```
index.html              home page: bio + thumbnail grid of projects
style.css               single shared stylesheet
.nojekyll               tells GitHub Pages to serve files as-is (skip Jekyll)
projects/
  mandelbrot/index.html   one folder per project, each with its own landing page
  duffing/index.html
  sandpile/index.html
  _template/index.html    copy this folder to start a new project page
assets/
  img/                  thumbnails and figures (one PNG per project, named after it)
  code/                 raw .py scripts served by the "download" links
```

## Publishing on GitHub Pages

1. Create a new repository on github.com. If you name it
   `<your-username>.github.io`, the site will live at `https://<your-username>.github.io/`.
   Any other name works too — the site then lives at
   `https://<your-username>.github.io/<repo-name>/` (all links in this site are
   relative, so both work).
2. From inside this `website/` folder:

   ```
   git init
   git add .
   git commit -m "Initial site"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

3. On github.com, open the repository → **Settings → Pages**, set
   **Source: Deploy from a branch**, branch `main`, folder `/ (root)`, and save.
   The site appears at the URL shown there within a minute or two.

To update the site later: edit files, then `git add . ; git commit -m "..." ; git push`.

## Adding a new project

1. Copy `projects/_template/` to `projects/<short-name>/` and fill in the
   `EDIT ME` spots (title, description, figure, code).
2. Put a thumbnail PNG in `assets/img/<short-name>.png` (roughly square or 4:3;
   the grid crops to 4:3 automatically).
3. Put the script in `assets/code/` and list it in the page's "Source code"
   download list (one `<li>` per file).
4. Add a card to the grid in `index.html` — there is a commented-out example
   card showing exactly what to paste.

## Notes

- Math is rendered by MathJax (loaded from a CDN), which requires internet
  access when viewing the page: `$...$` inline, `$$...$$` display.
- Source code is offered as download links only (no inline listings on the pages).
- Viewing locally: just open `index.html` in a browser — everything works from
  the filesystem, no server needed.
- LaTeX project write-ups: for a future project based on a `.tex` paper, the
  simplest route is to put the compiled PDF in `assets/` and link it from the
  project page, copying the key equations into the HTML as MathJax.
