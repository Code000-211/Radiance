import assert from "node:assert/strict";
import { appendFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import type { Chunk } from "../src/chunks.ts";
import { MemoryStore, type ArchiveRecord } from "../src/store.ts";

function chunk(id: string, kind: Chunk["kind"], text: string, archiveId?: string): Chunk {
    return { id, entryId: id.split("#")[0], kind, text, timestamp: 1, archiveId };
}

function record(id: string, summary: string, radiance = true): ArchiveRecord {
    return { id, sessionId: "s", createdAt: 1, firstKeptEntryId: "k", tokensBefore: 10, summary, radiance };
}

test("the store survives a reload, a torn last line included", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-memory-"));
    try {
        const s = new MemoryStore(dir);
        await s.load();
        await s.addChunks([chunk("e1", "user", "rename the fork endpoint", "A"),
                           chunk("e2", "tool", "build log: 672 targets", "A")], [[1, 0], [0, 1]]);
        await s.addChunks([chunk("e1", "user", "a repeat is not added")]);
        await s.addArchive(record("A", "renamed the endpoint"));
        await appendFile(join(dir, "chunks.jsonl"), '{"id":"torn"');

        const t = new MemoryStore(dir);
        await t.load();
        assert.equal(t.size, 2);
        assert.ok(t.has("e1") && !t.has("torn"));
        assert.equal(t.archive("A")?.summary, "renamed the endpoint");
        assert.equal(t.search("fork endpoint", 5)[0].chunk.id, "e1");
        /* The semantic half alone finds what shares no word with the query. */
        assert.equal(t.search("zzz", 5, [0, 1])[0].chunk.id, "e2");
    } finally {
        await rm(dir, { recursive: true, force: true });
    }
});

test("decisions outrank tool output, and an archive whose summary matches lifts its chunks", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-memory-"));
    try {
        const s = new MemoryStore(dir);
        await s.load();
        await s.addChunks([chunk("t", "tool", "snapshot quantum is 64", "A"),
                           chunk("d", "decision", "snapshot quantum is 64", "B")]);
        assert.equal(s.search("snapshot quantum", 2)[0].chunk.id, "d");

        await s.addChunks([chunk("u1", "user", "pinning the cache", "C"), chunk("u2", "user", "pinning the cache", "D")]);
        await s.addArchive(record("C", "unrelated"));
        await s.addArchive(record("D", "everything about pinning"));
        assert.equal(s.search("pinning", 1)[0].chunk.id, "u2");
        assert.equal(s.archiveFor("pinning")?.id, "D");
    } finally {
        await rm(dir, { recursive: true, force: true });
    }
});

test("a forgotten archive is text only, and stays so after a reload", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-memory-"));
    try {
        const s = new MemoryStore(dir);
        await s.load();
        await s.addArchive(record("A", "first"));
        await s.addArchive(record("L", "local only", false));
        assert.equal(s.archiveFor("anything")?.id, "A");
        await s.forget("A");
        assert.equal(s.archive("A")?.radiance, false);
        assert.equal(s.archiveFor("anything"), undefined);

        const t = new MemoryStore(dir);
        await t.load();
        assert.equal(t.archives().length, 2);
        assert.equal(t.archive("A")?.radiance, false);
    } finally {
        await rm(dir, { recursive: true, force: true });
    }
});
