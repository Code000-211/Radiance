/* The runtime against a stand-in for Radiance's /v1/memory routes (core/server/memory.cpp). */
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { createServer, type IncomingMessage, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, beforeEach, test } from "node:test";

import type {
    ExtensionContext,
    SessionBeforeCompactEvent,
    SessionCompactEvent,
    SessionEntry,
} from "@earendil-works/pi-coding-agent";

import { configFromEnv, MemoryRuntime } from "../src/memory.ts";

interface Seen {
    method: string;
    path: string;
    auth?: string;
    body: Record<string, unknown>;
}

/* Radiance as the routes answer, holding archives by id and at most `limit` of them. */
class FakeRadiance {
    server: Server | undefined;
    base = "";
    seen: Seen[] = [];
    archives = new Map<string, { session: string; age: number }>();
    limit = 8;
    forkStatus = 200;
    summaryFinish = "stop";
    memory = true;

    async start(): Promise<void> {
        this.server = createServer(async (req, res) => {
            const body = await read(req);
            const path = req.url ?? "";
            this.seen.push({ method: req.method ?? "", path, auth: req.headers.authorization, body });
            const reply = (status: number, j: unknown) => {
                res.writeHead(status, { "Content-Type": "application/json" });
                res.end(JSON.stringify(j));
            };
            const fail = (status: number, message: string) => reply(status, { error: { message } });
            if (!this.memory) return fail(404, "no such route");
            if (req.method === "GET" && path === "/v1/memory/archive")
                return reply(200, { object: "list", data: [...this.archives].map(([id, a]) => stat(id, a)) });
            if (req.method === "POST" && path === "/v1/memory/archive") {
                const id = String(body.archive_id);
                if (this.archives.has(id)) return fail(409, "exists");
                if (this.archives.size >= this.limit) return fail(507, "full");
                this.archives.set(id, { session: String(body.session_id), age: 0 });
                return reply(201, stat(id, this.archives.get(id)!));
            }
            if (req.method === "DELETE" && path.startsWith("/v1/memory/archive/")) {
                const id = decodeURIComponent(path.slice("/v1/memory/archive/".length));
                if (!this.archives.delete(id)) return fail(404, `no archive '${id}'`);
                return reply(200, { id, deleted: true });
            }
            if (req.method === "POST" && path === "/v1/memory/fork") {
                if (this.forkStatus !== 200) return fail(this.forkStatus, "the engine is down");
                const src = body.source as { archive?: string; session?: string };
                if (src.archive && !this.archives.has(src.archive)) return fail(404, `no archive '${src.archive}'`);
                const answers = (body.questions as string[]).map((q, index) => {
                    if (q.includes("Write it in exactly this format"))
                        return { index, question: q, content: "## Goal\nShip the memory routes.",
                                 finish_reason: this.summaryFinish };
                    if (q.includes("List every decision"))
                        return { index, question: q, finish_reason: "stop",
                                 content: "- DECISION: pin by hash | WHY: survives recompute | REJECTED: by block id | FILES: core/mem/prefix.cpp\n"
                                     + "- DECISION: one snapshot per archive | WHY: 113 MiB each | REJECTED: all | FILES: none" };
                    return { index, question: q, content: "Because pins are keyed by hash.", finish_reason: "stop" };
                });
                return reply(200, { object: "memory.fork", source: src, source_tokens: 4096, answers,
                                    usage: { prompt_tokens: 4200, completion_tokens: 40, total_tokens: 4240,
                                             prompt_tokens_details: { cached_tokens: 4096 } } });
            }
            fail(404, "no such route");
        });
        await new Promise<void>((r) => this.server!.listen(0, "127.0.0.1", r));
        this.base = `http://127.0.0.1:${(this.server.address() as AddressInfo).port}`;
    }

    stop(): Promise<void> {
        return new Promise((r) => this.server!.close(() => r()));
    }

    calls(method: string, path: string): Seen[] {
        return this.seen.filter((s) => s.method === method && s.path === path);
    }
}

function stat(id: string, a: { session: string; age: number }) {
    return { id, object: "memory.archive", session_id: a.session, tokens: 4096, pinned_tokens: 4096,
             snapshot_tokens: 4096, age_s: a.age };
}

async function read(req: IncomingMessage): Promise<Record<string, unknown>> {
    let s = "";
    for await (const c of req) s += c;
    return s ? JSON.parse(s) : {};
}

function entry(id: string, message: Record<string, unknown>): SessionEntry {
    return { type: "message", id, parentId: null, timestamp: "2026-10-01T10:00:00.000Z", message } as unknown as SessionEntry;
}

const BRANCH: SessionEntry[] = [
    entry("e1", { role: "user", content: "the memory routes fail to link" }),
    entry("e2", { role: "assistant", content: [{ type: "thinking", thinking: "memory.cpp came after configure" },
                                               { type: "text", text: "Reconfiguring fixes the linker." }] }),
    entry("e3", { role: "user", content: "now the tests" }),
];

function context(base: string, api = "openai-completions", sid = "sess-1") {
    const notes: string[] = [];
    const branch = [...BRANCH];
    const ctx = {
        sessionManager: { getSessionId: () => sid, getBranch: () => branch },
        model: { baseUrl: `${base}/v1`, api },
        modelRegistry: { getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "sk-test" }) },
        ui: { notify: (m: string) => notes.push(m) },
    } as unknown as ExtensionContext;
    return { ctx, notes, branch };
}

function beforeCompact(): SessionBeforeCompactEvent {
    return {
        type: "session_before_compact",
        preparation: {
            firstKeptEntryId: "e3",
            messagesToSummarize: [],
            turnPrefixMessages: [],
            isSplitTurn: false,
            tokensBefore: 5000,
            fileOps: { read: new Set(["core/server/memory.cpp", "core/mem/prefix.cpp"]),
                       written: new Set(["core/mem/prefix.cpp"]), edited: new Set() },
            settings: { enabled: true, reserveTokens: 16384, keepRecentTokens: 20000 },
        },
        branchEntries: BRANCH,
        reason: "threshold",
        willRetry: false,
        signal: new AbortController().signal,
    } as unknown as SessionBeforeCompactEvent;
}

function compacted(summary: string, branch: SessionEntry[]): SessionCompactEvent {
    const compactionEntry = { type: "compaction", id: "c1", parentId: "e2", timestamp: "2026-10-01T11:00:00.000Z",
                              summary, firstKeptEntryId: "e3", tokensBefore: 5000 };
    branch.push(compactionEntry as unknown as SessionEntry);
    return { type: "session_compact", compactionEntry, fromExtension: true, reason: "threshold",
             willRetry: false } as unknown as SessionCompactEvent;
}

const radiance = new FakeRadiance();
let dir = "";
let rt: MemoryRuntime;

before(() => radiance.start());
after(() => radiance.stop());
beforeEach(async () => {
    if (dir) await rm(dir, { recursive: true, force: true });
    dir = await mkdtemp(join(tmpdir(), "pi-memory-rt-"));
    rt = new MemoryRuntime({ ...configFromEnv({}), dir });
    radiance.seen = [];
    radiance.archives.clear();
    radiance.limit = 8;
    radiance.forkStatus = 200;
    radiance.summaryFinish = "stop";
    radiance.memory = true;
});
after(() => rm(dir, { recursive: true, force: true }));

test("requests to a Radiance model are named by the session, others are left alone", async () => {
    const { ctx } = context(radiance.base);
    assert.deepEqual(await rt.tag({ model: "m", messages: [] }, ctx), { model: "m", messages: [], session_id: "sess-1" });
    const { ctx: anthropic } = context(radiance.base, "anthropic-messages");
    assert.deepEqual(await rt.tag({ metadata: { x: 1 } }, anthropic), { metadata: { x: 1, user_id: "sess-1" } });
    assert.equal(radiance.calls("GET", "/v1/memory/archive")[0].auth, "Bearer sk-test");

    const other = new FakeRadiance();
    other.memory = false;
    await other.start();
    try {
        assert.equal(await rt.tag({ model: "m" }, context(other.base).ctx), undefined);
    } finally {
        await other.stop();
    }
});

test("compaction forks the summary and decisions, archives, and indexes what left the context", async () => {
    const { ctx, branch } = context(radiance.base);
    const r = await rt.compact(beforeCompact(), ctx);
    assert.ok(r?.compaction);
    const c = r.compaction;
    assert.match(c.summary, /^## Goal\nShip the memory routes\./);
    assert.match(c.summary, /<read-files>\ncore\/server\/memory\.cpp\n<\/read-files>/);
    assert.match(c.summary, /<modified-files>\ncore\/mem\/prefix\.cpp\n<\/modified-files>/);
    assert.equal(c.firstKeptEntryId, "e3");
    assert.equal(c.tokensBefore, 5000);

    const [fork] = radiance.calls("POST", "/v1/memory/fork");
    assert.deepEqual(fork.body.source, { session: "sess-1" });
    assert.equal(fork.body.checkpoint_at_end, true);
    assert.equal((fork.body.questions as string[]).length, 2);
    assert.match((fork.body.questions as string[])[0], /roughly the last 20000 tokens/);
    const [arc] = radiance.calls("POST", "/v1/memory/archive");
    assert.equal(arc.body.session_id, "sess-1");
    const archiveId = String(arc.body.archive_id);
    assert.deepEqual(c.details, { radianceArchive: archiveId });

    await rt.indexCompaction(compacted(c.summary, branch), ctx);
    const hits = await rt.search("linker", 5, ctx);
    assert.equal(hits[0].chunk.entryId, "e2");
    assert.equal(hits[0].chunk.archiveId, archiveId);
    const decision = await rt.search("pin by hash", 1, ctx);
    assert.equal(decision[0].chunk.kind, "decision");
    assert.equal((await rt.search("now the tests", 5, ctx)).filter((h) => h.chunk.entryId === "e3").length, 0);

    const a = await rt.ask("why are pins keyed by hash?", undefined, ctx);
    assert.equal(a.text, "Because pins are keyed by hash.");
    assert.equal(a.archive?.id, archiveId);
    assert.equal(a.cached, 4096);
    assert.deepEqual(radiance.calls("POST", "/v1/memory/fork")[1].body.source, { archive: archiveId });
    assert.match(await rt.describe(ctx), new RegExp(`${archiveId} .* pinned, 4096 tokens`));
});

test("a failed fork or a cut-off summary leaves the compaction to Pi", async () => {
    const { ctx, notes } = context(radiance.base);
    radiance.forkStatus = 503;
    assert.equal(await rt.compact(beforeCompact(), ctx), undefined);
    assert.match(notes[0], /the fork failed \(503: the engine is down\)/);

    radiance.forkStatus = 200;
    radiance.summaryFinish = "length";
    assert.equal(await rt.compact(beforeCompact(), ctx), undefined);
    assert.equal(radiance.calls("POST", "/v1/memory/archive").length, 0);
});

test("a full server gives up this client's oldest archive, never another client's", async () => {
    const { ctx } = context(radiance.base);
    radiance.limit = 2;
    radiance.archives.set("pi-older", { session: "s0", age: 900 });
    radiance.archives.set("other-client", { session: "x", age: 5000 });
    const r = await rt.compact(beforeCompact(), ctx);
    assert.ok(r?.compaction?.details);
    assert.ok(!radiance.archives.has("pi-older"));
    assert.ok(radiance.archives.has("other-client"));
    assert.equal(radiance.archives.size, 2);

    radiance.archives.clear();
    radiance.archives.set("a", { session: "x", age: 1 });
    radiance.archives.set("b", { session: "x", age: 2 });
    const { ctx: c2, notes } = context(radiance.base);
    const r2 = await rt.compact(beforeCompact(), c2);
    assert.deepEqual(r2?.compaction?.details, { radianceArchive: undefined });
    assert.match(notes.join("\n"), /not archived .*another client's/);
});

test("an archive the server let go is answered from its text", async () => {
    const { ctx, branch } = context(radiance.base);
    const r = await rt.compact(beforeCompact(), ctx);
    await rt.indexCompaction(compacted(r!.compaction!.summary, branch), ctx);
    radiance.archives.clear();
    const a = await rt.ask("why?", undefined, ctx);
    assert.match(a.text, /no longer held by the server/);
    assert.match((await rt.ask("why?", undefined, ctx)).text, /Nothing in this session has been archived/);
    assert.equal((await rt.search("linker", 1, ctx)).length, 1);
});

test("without Radiance, Pi compacts and the text is still indexed", async () => {
    radiance.memory = false;
    const { ctx, branch } = context(radiance.base);
    assert.equal(await rt.compact(beforeCompact(), ctx), undefined);
    await rt.indexCompaction(compacted("## Goal\nPi's own summary", branch), ctx);
    const hits = await rt.search("linker", 1, ctx);
    assert.equal(hits[0].chunk.archiveId, "local-c1");
    assert.match((await rt.ask("why?", undefined, ctx)).text, /not served by a Radiance server/);
});
