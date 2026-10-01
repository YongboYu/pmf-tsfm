// Write slides.public.md: slides.md minus speaker notes and backup slides. Used by `pnpm build:public`
// (the HF Space deploy). The generated file is gitignored.
import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { isBackup, splitSlides, toPublicDeck } from './publicDeck.js'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const src = readFileSync(join(root, 'slides.md'), 'utf8')
const out = toPublicDeck(src)
writeFileSync(join(root, 'slides.public.md'), out)

const total = splitSlides(src).length
const dropped = splitSlides(src).filter(isBackup).length
console.log(`slides.public.md: ${total - dropped} slides (dropped ${dropped} backup), notes stripped`)
