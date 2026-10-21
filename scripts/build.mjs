import { spawn } from 'node:child_process'
import { readdir, readFile, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const slidesRoot = path.join(root, 'slides')
const distRoot = path.join(root, 'dist')
const slidevBin = path.join(root, 'node_modules', '.bin', 'slidev')

function normalizeBase(input) {
  const raw = (input ?? '/').trim() || '/'
  const withLeadingSlash = raw.startsWith('/') ? raw : `/${raw}`
  return withLeadingSlash.endsWith('/') ? withLeadingSlash : `${withLeadingSlash}/`
}

function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
}

function readTitle(markdown, fallback) {
  const fence = markdown.match(/^---\r?\n([\s\S]*?)\r?\n---/)
  if (!fence)
    return fallback
  const title = fence[1].match(/^title:\s*(.+?)\s*$/m)
  if (!title)
    return fallback
  return title[1].replace(/^['"]|['"]$/g, '')
}

function runSlidev(args) {
  return new Promise((resolve, reject) => {
    const child = spawn(slidevBin, args, { cwd: root, stdio: 'inherit' })
    child.on('error', reject)
    child.on('exit', (code) => {
      if (code === 0)
        resolve()
      else
        reject(new Error(`slidev exited with code ${code}`))
    })
  })
}

async function listDecks() {
  let entries
  try {
    entries = await readdir(slidesRoot, { withFileTypes: true })
  }
  catch (error) {
    if (error.code === 'ENOENT')
      return []
    throw error
  }

  const decks = []
  for (const entry of entries) {
    if (!entry.isDirectory())
      continue
    const file = path.join(slidesRoot, entry.name, 'slides.md')
    let markdown
    try {
      markdown = await readFile(file, 'utf8')
    }
    catch (error) {
      if (error.code === 'ENOENT')
        continue
      throw error
    }
    decks.push({
      name: entry.name,
      file,
      title: readTitle(markdown, entry.name),
    })
  }
  decks.sort((a, b) => a.name.localeCompare(b.name))
  return decks
}

const basePath = normalizeBase(process.env.BASE_PATH)
const decks = await listDecks()
if (decks.length === 0) {
  console.error('No decks found. Add slides/<name>/slides.md')
  process.exit(1)
}

await rm(distRoot, { recursive: true, force: true })

for (const deck of decks) {
  const outDir = path.join(distRoot, deck.name)
  const base = `${basePath}${deck.name}/`
  console.log(`\nBuilding ${deck.name} -> ${path.relative(root, outDir)} (base ${base})\n`)
  await runSlidev([
    'build',
    deck.file,
    '--out',
    outDir,
    '--base',
    base,
    '--router-mode',
    'hash',
  ])
}

const items = decks.map((deck) => {
  const href = `./${encodeURIComponent(deck.name)}/`
  return `      <li><a href="${href}"><span class="title">${escapeHtml(deck.title)}</span><span class="path">${escapeHtml(deck.name)}</span></a></li>`
}).join('\n')

const html = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Slides</title>
  <style>
    :root { color-scheme: light dark; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; }
    main { max-width: 40rem; margin: 0 auto; padding: 4rem 1.5rem; }
    h1 { font-size: 1.75rem; font-weight: 600; margin: 0 0 0.5rem; }
    p { margin: 0 0 2rem; opacity: 0.7; }
    ul { list-style: none; padding: 0; margin: 0; }
    a {
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      padding: 0.9rem 0;
      border-top: 1px solid color-mix(in srgb, currentColor 15%, transparent);
      color: inherit;
      text-decoration: none;
    }
    li:last-child a { border-bottom: 1px solid color-mix(in srgb, currentColor 15%, transparent); }
    a:hover .title { text-decoration: underline; }
    .path {
      opacity: 0.55;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <main>
    <h1>Slides</h1>
    <p>${decks.length} 份演示文稿</p>
    <ul>
${items}
    </ul>
  </main>
</body>
</html>
`

await writeFile(path.join(distRoot, 'index.html'), html)
console.log(`\nWrote ${decks.length} deck(s) to dist/\n`)
