/* store.ts -- one Pi session's memory on disk: the chunks compaction removed from the live
 * context, the archives Radiance keeps of the conversations they came from, and search over both.
 *
 * Append-only JSON lines, read whole at session start: a session's history only grows, and a
 * crash mid-append loses at most the last line, which load() skips.
 *
 *   chunks.jsonl     Chunk, one a line
 *   vectors.jsonl    {id, v}: a chunk's embedding, when an embedder is configured
 *   archives.jsonl   ArchiveRecord, one a line, oldest first
 *
 * Search is two-level. The archives' summaries are ranked first, as a coarse index of which
 * stretch of the session a question is about; then chunks are ranked by BM25 and, with
 * embeddings, by cosine similarity, fused by rank, weighted by kind, and boosted when they belong
 * to an archive whose summary matched. */
import { appendFile, mkdir, readFile } from "node:fs/promises";
import { join } from "node:path";

import type { Chunk, ChunkKind } from "./chunks.ts";
import { Bm25, cosine, fuse, type Scored } from "./text.ts";

export interface ArchiveRecord {
    id: string;                  /* the Radiance archive id, or a local one when there is none */
    sessionId: string;
    createdAt: number;
    firstKeptEntryId: string;
    tokensBefore: number;
    summary: string;
    /* Whether Radiance pinned the conversation: memory_ask can question it. Without, only its
     * text was kept. */
    radiance: boolean;
    sourceTokens?: number;
}

export interface Hit {
    chunk: Chunk;
    score: number;
}

/* Tool output is the bulk of a session and mostly reproducible; a decision record is what a
 * question about the past most often wants. */
const KIND_WEIGHT: Record<ChunkKind, number> = {
    decision: 1.5,
    user: 1.2,
    assistant: 1.0,
    summary: 1.0,
    tool: 0.6,
};
const ARCHIVE_BOOST = 1.3;

async function readLines(path: string): Promise<unknown[]> {
    let text: string;
    try {
        text = await readFile(path, "utf8");
    } catch (e) {
        if ((e as NodeJS.ErrnoException).code === "ENOENT") return [];
        throw e;
    }
    const out: unknown[] = [];
    for (const line of text.split("\n")) {
        if (!line.trim()) continue;
        try {
            out.push(JSON.parse(line));
        } catch {
            /* A torn last line from a crash mid-append. */
        }
    }
    return out;
}

export class MemoryStore {
    readonly dir: string;
    private readonly chunks = new Map<string, Chunk>();
    private readonly entries = new Set<string>();
    private readonly vectors = new Map<string, number[]>();
    private readonly lexical = new Bm25();
    private readonly summaries = new Bm25();
    private readonly records: ArchiveRecord[] = [];

    constructor(dir: string) {
        this.dir = dir;
    }

    async load(): Promise<void> {
        await mkdir(this.dir, { recursive: true });
        for (const c of (await readLines(join(this.dir, "chunks.jsonl"))) as Chunk[]) this.take(c);
        for (const v of (await readLines(join(this.dir, "vectors.jsonl"))) as Array<{ id: string; v: number[] }>)
            if (v && typeof v.id === "string" && Array.isArray(v.v)) this.vectors.set(v.id, v.v);
        for (const r of (await readLines(join(this.dir, "archives.jsonl"))) as ArchiveRecord[])
            if (r && typeof r.id === "string") this.keep(r);
    }

    private take(c: Chunk): void {
        if (!c || typeof c.id !== "string" || this.chunks.has(c.id)) return;
        this.chunks.set(c.id, c);
        this.entries.add(c.entryId);
        this.lexical.add(c.id, c.text);
    }

    private keep(r: ArchiveRecord): void {
        const i = this.records.findIndex((x) => x.id === r.id);
        if (i >= 0) this.records[i] = r;
        else this.records.push(r);
        this.summaries.add(r.id, r.summary);
    }

    /* Whether an entry's chunks are already held, so a second pass over a branch adds nothing. */
    has(entryId: string): boolean {
        return this.entries.has(entryId);
    }

    get size(): number {
        return this.chunks.size;
    }

    archives(): readonly ArchiveRecord[] {
        return this.records;
    }

    archive(id: string): ArchiveRecord | undefined {
        return this.records.find((r) => r.id === id);
    }

    async addChunks(chunks: Chunk[], vectors?: number[][]): Promise<void> {
        const fresh = chunks.filter((c) => !this.chunks.has(c.id));
        if (fresh.length === 0) return;
        await appendFile(join(this.dir, "chunks.jsonl"),
                         fresh.map((c) => JSON.stringify(c)).join("\n") + "\n");
        for (const c of fresh) this.take(c);
        if (!vectors || vectors.length !== chunks.length) return;
        const lines: string[] = [];
        chunks.forEach((c, i) => {
            if (this.vectors.has(c.id)) return;
            this.vectors.set(c.id, vectors[i]);
            lines.push(JSON.stringify({ id: c.id, v: vectors[i] }));
        });
        if (lines.length) await appendFile(join(this.dir, "vectors.jsonl"), lines.join("\n") + "\n");
    }

    async addArchive(r: ArchiveRecord): Promise<void> {
        await appendFile(join(this.dir, "archives.jsonl"), JSON.stringify(r) + "\n");
        this.keep(r);
    }

    /* An archive the server no longer holds: its text stays searchable, memory_ask passes it by. */
    async forget(id: string): Promise<void> {
        const r = this.archive(id);
        if (!r || !r.radiance) return;
        await this.addArchive({ ...r, radiance: false });
    }

    /* The best `limit` chunks for `query`, best first. `queryVector` is the query's embedding when
     * an embedder is configured. */
    search(query: string, limit: number, queryVector?: number[]): Hit[] {
        const pool = Math.max(50, limit * 5);
        const lists: Scored[][] = [this.lexical.search(query, pool)];
        if (queryVector && this.vectors.size > 0) {
            const near: Scored[] = [];
            /* Only what points the query's way votes; rank fusion would count the rest. */
            for (const [id, v] of this.vectors) {
                const score = cosine(queryVector, v);
                if (score > 0) near.push({ id, score });
            }
            near.sort((a, b) => b.score - a.score);
            lists.push(near.slice(0, pool));
        }
        const fused = fuse(lists);
        const boosted = new Set(this.summaries.search(query, 2).map((s) => s.id));
        const hits: Hit[] = [];
        for (const [id, score] of fused) {
            const chunk = this.chunks.get(id);
            if (!chunk) continue;
            let s = score * KIND_WEIGHT[chunk.kind];
            if (chunk.archiveId && boosted.has(chunk.archiveId)) s *= ARCHIVE_BOOST;
            hits.push({ chunk, score: s });
        }
        hits.sort((a, b) => b.score - a.score || b.chunk.timestamp - a.chunk.timestamp);
        return hits.slice(0, limit);
    }

    /* The archive a question most likely concerns: the one most of its best chunks came from,
     * else the newest that Radiance pinned. */
    archiveFor(query: string, queryVector?: number[]): ArchiveRecord | undefined {
        const pinned = this.records.filter((r) => r.radiance);
        if (pinned.length === 0) return undefined;
        const votes = new Map<string, number>();
        for (const h of this.search(query, 12, queryVector))
            if (h.chunk.archiveId) votes.set(h.chunk.archiveId, (votes.get(h.chunk.archiveId) ?? 0) + h.score);
        let best: ArchiveRecord | undefined;
        let bestVotes = 0;
        for (const r of pinned) {
            const v = votes.get(r.id) ?? 0;
            if (v > bestVotes) {
                best = r;
                bestVotes = v;
            }
        }
        return best ?? pinned[pinned.length - 1];
    }
}
