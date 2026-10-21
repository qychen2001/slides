import { spawn } from 'node:child_process'
import { readdir } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const slidevBin = path.join(root, 'node_modules', '.bin', 'slidev')
const [name, ...rest] = process.argv.slice(2)

async function listDecks() {
  const entries = await readdir(path.join(root, 'slides'), { withFileTypes: true })
  return entries.filter(entry => entry.isDirectory()).map(entry => entry.name).sort()
}

if (!name) {
  const decks = await listDecks().catch(() => [])
  console.error('Usage: npm run dev -- <deck-name>')
  console.error('Example: npm run dev -- welcome')
  if (decks.length)
    console.error(`Decks: ${decks.join(', ')}`)
  process.exit(1)
}

const entry = name.endsWith('.md') || name.includes('/')
  ? name
  : path.join('slides', name, 'slides.md')

const child = spawn(slidevBin, [entry, '--open', ...rest], {
  cwd: root,
  stdio: 'inherit',
})
child.on('error', (error) => {
  console.error(error)
  process.exit(1)
})
child.on('exit', code => process.exit(code ?? 1))
