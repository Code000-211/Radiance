/* memory.ts -- the extension's state for one Pi session, and what each hook and tool does with it.
 *
 * Radiance keeps a conversation's KV as the agent left it; compaction is when Pi stops sending
 * it. At that moment this asks Radiance to fork the conversation for its summary (a prefix-cache
 * hit, so only the summary is computed) and to archive it (pin its cached chain and the snapshot
 * the fork wrote at its end), and indexes the messages that leave the live context. Later the
 * agent searches that text (memory_search) or questions the archived conversation itself
 * (memory_ask), which answers from the whole of it.
 *
 * Every Radiance step can fail without harm: compaction then falls back to Pi's own, and the
 * text index is kept either way. Pi's session file stays the record of what was said; the
 * archive is a fast way back into it, not a copy that must survive. */
import { join } from "node:path";
import { homedir } from "node:os";

import type {
    ExtensionContext,
    SessionBeforeCompactEvent,
    SessionBeforeCompactResult,
    SessionCompactEvent,
    SessionEntry,
} from "@earendil-works/pi-coding-agent";

import { chunksOf, type Chunk } from "./chunks.ts";
import { DECISIONS_QUESTION, decisionLines, fileLists, recallQuestion, summaryQuestion } from "./prompts.ts";
import { Embedder, RadianceError, RadianceMemory, type ArchiveStat } from "./radiance.ts";
import { MemoryStore, type ArchiveRecord, type Hit } from "./store.ts";

export interface Config {
    url?: string;               /* RADIANCE_MEMORY_URL: the server's root; else the model's */
    key?: string;               /* RADIANCE_MEMORY_KEY; else the model's API key */
    dir: string;                /* RADIANCE_MEMORY_DIR */
    embedUrl?: string;          /* RADIANCE_MEMORY_EMBED_URL, RADIANCE_MEMORY_EMBED_MODEL, */
    embedModel?: string;        /* RADIANCE_MEMORY_EMBED_KEY: an OpenAI-compatible endpoint */
    embedKey?: string;
    summaryThinking: number;    /* reasoning tokens the summary may spend */
    summaryMaxTokens: number;
    askThinking: number;        /* ... and a recall answer */
    askMaxTokens: number;
    decisions: boolean;         /* also ask for decision records at compaction */
}

function int(v: string | undefined, d: number): number {
    const n = v === undefined ? NaN : Number.parseInt(v, 10);
    return Number.isFinite(n) && n >= 0 ? n : d;
}

export function configFromEnv(env: NodeJS.ProcessEnv = process.env): Config {
    return {
        url: env.RADIANCE_MEMORY_URL || undefined,
        key: env.RADIANCE_MEMORY_KEY || undefined,
        dir: env.RADIANCE_MEMORY_DIR || join(homedir(), ".pi", "agent", "radiance-memory"),
        embedUrl: env.RADIANCE_MEMORY_EMBED_URL || undefined,
        embedModel: env.RADIANCE_MEMORY_EMBED_MODEL || undefined,
        embedKey: env.RADIANCE_MEMORY_EMBED_KEY || undefined,
        summaryThinking: int(env.RADIANCE_MEMORY_SUMMARY_THINKING, 1024),
        summaryMaxTokens: int(env.RADIANCE_MEMORY_SUMMARY_MAX_TOKENS, 8192),
        askThinking: int(env.RADIANCE_MEMORY_ASK_THINKING, 512),
        askMaxTokens: int(env.RADIANCE_MEMORY_ASK_MAX_TOKENS, 2048),
        decisions: env.RADIANCE_MEMORY_DECISIONS !== "0",
    };
}

/* The server root an OpenAI-style base URL belongs to: http://host:8000/v1 -> http://host:8000. */
export function serverRoot(baseUrl: string): string {
    return baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "");
}

/* A session id as a directory name. */
function safe(id: string): string {
    return id.replace(/[^A-Za-z0-9._-]/g, "_");
}

/* What a compaction this extension wrote leaves for session_compact, which indexes it. */
interface Pending {
    archiveId: string;
    radiance: boolean;
    firstKeptEntryId: string;
    decisions: string[];
    sourceTokens?: number;
}

export class MemoryRuntime {
    readonly cfg: Config;
    private store: MemoryStore | undefined;
    private storeFor = "";
    private readonly embedder: Embedder | undefined;
    /* Whether a server root keeps memory for this client, asked once per root. */
    private readonly probed = new Map<string, Promise<boolean>>();
    private pending: Pending | undefined;

    constructor(cfg: Config) {
        this.cfg = cfg;
        this.embedder = cfg.embedUrl && cfg.embedModel
            ? new Embedder(cfg.embedUrl, cfg.embedModel, cfg.embedKey) : undefined;
    }

    /* The session's store, opened on first use and again after a session switch. */
    async storeOf(ctx: ExtensionContext): Promise<MemoryStore> {
        const sid = ctx.sessionManager.getSessionId();
        if (!this.store || this.storeFor !== sid) {
            const s = new MemoryStore(join(this.cfg.dir, safe(sid)));
            await s.load();
            this.store = s;
            this.storeFor = sid;
            this.pending = undefined;
        }
        return this.store;
    }

    /* The Radiance server the current model is served by, when it keeps memory for us. */
    async server(ctx: ExtensionContext, signal?: AbortSignal): Promise<RadianceMemory | undefined> {
        const base = this.cfg.url ?? (ctx.model?.baseUrl ? serverRoot(ctx.model.baseUrl) : undefined);
        if (!base) return undefined;
        let key = this.cfg.key;
        if (!key && ctx.model && serverRoot(ctx.model.baseUrl) === serverRoot(base)) {
            const auth = await ctx.modelRegistry.getApiKeyAndHeaders(ctx.model);
            if (auth.ok) key = auth.apiKey;
        }
        const client = new RadianceMemory(base, key);
        let ok = this.probed.get(client.base);
        if (!ok) {
            ok = client.available(signal);
            this.probed.set(client.base, ok);
        }
        return (await ok) ? client : undefined;
    }

    /* before_provider_request: name the conversation, so Radiance keeps its tokens under the
     * session id. Only for a model Radiance serves: another provider may refuse a field it does
     * not know. */
    async tag(payload: unknown, ctx: ExtensionContext): Promise<unknown> {
        if (!payload || typeof payload !== "object" || !ctx.model) return undefined;
        const server = await this.server(ctx);
        if (!server || serverRoot(ctx.model.baseUrl) !== server.base) return undefined;
        const sid = ctx.sessionManager.getSessionId();
        const p = payload as Record<string, unknown>;
        if (ctx.model.api === "anthropic-messages") {
            /* Radiance reads Anthropic's metadata.user_id as the label. */
            const meta = (p.metadata && typeof p.metadata === "object" ? p.metadata : {}) as Record<string, unknown>;
            return { ...p, metadata: { ...meta, user_id: sid } };
        }
        return { ...p, session_id: sid };
    }

    /* session_before_compact: the summary from a fork of the cached conversation, and an archive
     * of it. Undefined leaves the compaction to Pi. */
    async compact(event: SessionBeforeCompactEvent, ctx: ExtensionContext): Promise<SessionBeforeCompactResult | undefined> {
        const { preparation, signal, customInstructions } = event;
        await this.storeOf(ctx);
        const server = await this.server(ctx, signal);
        if (!server) return undefined;
        const sid = ctx.sessionManager.getSessionId();

        const questions = [summaryQuestion(preparation.settings.keepRecentTokens, customInstructions)];
        if (this.cfg.decisions) questions.push(DECISIONS_QUESTION);
        let summary = "";
        let decisions: string[] = [];
        let sourceTokens: number | undefined;
        try {
            const fork = await server.fork({ session: sid }, questions, {
                checkpointAtEnd: true,
                maxTokens: this.cfg.summaryMaxTokens,
                thinkingBudget: this.cfg.summaryThinking,
                signal,
            });
            summary = fork.answers[0]?.content?.trim() ?? "";
            if (fork.answers[0]?.finish_reason === "length") summary = "";
            decisions = decisionLines(fork.answers[1]?.content ?? "");
            sourceTokens = fork.source_tokens;
        } catch (e) {
            if (!signal.aborted)
                ctx.ui.notify(`radiance memory: the fork failed (${(e as Error).message}); Pi compacts instead`,
                              "warning");
            return undefined;
        }
        if (!summary) {
            ctx.ui.notify("radiance memory: the fork wrote no complete summary; Pi compacts instead", "warning");
            return undefined;
        }

        const archiveId = `pi-${safe(sid).slice(0, 12)}-${Date.now().toString(36)}`;
        let radiance = false;
        try {
            await this.archive(server, sid, archiveId, signal);
            radiance = true;
        } catch (e) {
            ctx.ui.notify(`radiance memory: not archived (${(e as Error).message}); the text is still indexed`,
                          "warning");
        }
        this.pending = { archiveId, radiance, firstKeptEntryId: preparation.firstKeptEntryId, decisions,
                         sourceTokens };
        return {
            compaction: {
                summary: summary + fileLists(preparation.fileOps),
                firstKeptEntryId: preparation.firstKeptEntryId,
                tokensBefore: preparation.tokensBefore,
                details: { radianceArchive: radiance ? archiveId : undefined },
            },
        };
    }

    /* An archive, making room when the server holds as many as it keeps (507): the oldest one this
     * extension wrote goes, its text staying searchable. Archives other clients wrote are left. */
    private async archive(server: RadianceMemory, sid: string, id: string, signal: AbortSignal): Promise<void> {
        try {
            await server.archive(sid, id, signal);
            return;
        } catch (e) {
            if (!(e instanceof RadianceError) || e.status !== 507) throw e;
        }
        const ours = (await server.archives(signal)).filter((a) => a.id.startsWith("pi-"));
        if (ours.length === 0) throw new RadianceError(507, "the server's archives are all another client's");
        const oldest = ours.reduce((a, b) => (b.age_s > a.age_s ? b : a));
        await server.remove(oldest.id, signal);
        await this.store?.forget(oldest.id);
        await server.archive(sid, id, signal);
    }

    /* What /memory shows: this session's archives, and what the server still holds of them. */
    async describe(ctx: ExtensionContext): Promise<string> {
        const store = await this.storeOf(ctx);
        const server = await this.server(ctx);
        let held = new Map<string, ArchiveStat>();
        if (server) {
            try {
                held = new Map((await server.archives()).map((a) => [a.id, a]));
            } catch {
                /* Listed as text-only below. */
            }
        }
        const lines = [`radiance memory: ${store.size} chunks indexed, ${store.archives().length} compactions`
                       + (server ? ` (server ${server.base})` : " (no Radiance server)")];
        for (const r of store.archives()) {
            const a = held.get(r.id);
            const when = new Date(r.createdAt).toISOString().replace("T", " ").slice(0, 16);
            const state = a ? `pinned, ${a.tokens} tokens, snapshot at ${a.snapshot_tokens}`
                    + (a.on_disk ? ", on disk" : "")
                : r.radiance ? "released (text only)" : "text only";
            lines.push(`  ${r.id}  ${when}  ${state}`);
        }
        return lines.join("\n");
    }

    /* session_compact: index what left the live context, whoever wrote the summary. */
    async indexCompaction(event: SessionCompactEvent, ctx: ExtensionContext): Promise<void> {
        const store = await this.storeOf(ctx);
        const entry = event.compactionEntry;
        const mine = this.pending && this.pending.firstKeptEntryId === entry.firstKeptEntryId
            ? this.pending : undefined;
        this.pending = undefined;
        const archiveId = mine?.archiveId ?? `local-${entry.id}`;

        const chunks: Chunk[] = [];
        for (const e of ctx.sessionManager.getBranch() as SessionEntry[]) {
            if (e.id === entry.firstKeptEntryId) break;
            if (e.id === entry.id || store.has(e.id)) continue;
            for (const c of chunksOf(e)) chunks.push({ ...c, archiveId });
        }
        for (const c of chunksOf(entry)) chunks.push({ ...c, archiveId });
        (mine?.decisions ?? []).forEach((d, i) => chunks.push({
            id: `${entry.id}#d${i}`,
            entryId: entry.id,
            kind: "decision",
            text: d,
            timestamp: Date.parse(entry.timestamp) || Date.now(),
            archiveId,
        }));

        let vectors: number[][] | undefined;
        if (this.embedder && chunks.length) {
            try {
                vectors = [];
                for (let i = 0; i < chunks.length; i += 64)
                    vectors.push(...(await this.embedder.embed(chunks.slice(i, i + 64).map((c) => c.text))));
            } catch (e) {
                vectors = undefined;
                ctx.ui.notify(`radiance memory: no embeddings (${(e as Error).message}); lexical search only`,
                              "warning");
            }
        }
        await store.addChunks(chunks, vectors);
        const record: ArchiveRecord = {
            id: archiveId,
            sessionId: ctx.sessionManager.getSessionId(),
            createdAt: Date.now(),
            firstKeptEntryId: entry.firstKeptEntryId,
            tokensBefore: entry.tokensBefore,
            summary: entry.summary,
            radiance: mine?.radiance ?? false,
            sourceTokens: mine?.sourceTokens,
        };
        await store.addArchive(record);
    }

    async search(query: string, limit: number, ctx: ExtensionContext, signal?: AbortSignal): Promise<Hit[]> {
        const store = await this.storeOf(ctx);
        let qv: number[] | undefined;
        if (this.embedder) {
            try {
                qv = (await this.embedder.embed([query], signal))[0];
            } catch {
                qv = undefined;          /* lexical only, as at indexing */
            }
        }
        return store.search(query, limit, qv);
    }

    /* memory_ask: the question asked of an archived conversation; the archive named, or the one
     * the question's best matches came from. */
    async ask(question: string, archiveId: string | undefined, ctx: ExtensionContext,
              signal?: AbortSignal): Promise<{ text: string; archive?: ArchiveRecord; cached?: number }> {
        const store = await this.storeOf(ctx);
        const server = await this.server(ctx, signal);
        if (!server)
            return { text: "The model is not served by a Radiance server with /v1/memory, so archived "
                         + "conversations cannot be questioned. Use memory_search instead." };
        let archive: ArchiveRecord | undefined;
        if (archiveId) {
            archive = store.archive(archiveId);
            if (!archive) return { text: `No archive ${archiveId} in this session.` };
            if (!archive.radiance)
                return { text: `Archive ${archiveId} kept only its text; use memory_search on it.` };
        } else {
            let qv: number[] | undefined;
            if (this.embedder) {
                try {
                    qv = (await this.embedder.embed([question], signal))[0];
                } catch {
                    qv = undefined;
                }
            }
            archive = store.archiveFor(question, qv);
            if (!archive)
                return { text: "Nothing in this session has been archived yet: it has not been compacted "
                             + "on a Radiance server." };
        }
        let fork;
        try {
            fork = await server.fork({ archive: archive.id }, [recallQuestion(question)], {
                maxTokens: this.cfg.askMaxTokens,
                thinkingBudget: this.cfg.askThinking,
                signal,
            });
        } catch (e) {
            if (!(e instanceof RadianceError) || e.status !== 404) throw e;
            /* Released to make room, or lost with a restart the server did not persist through. */
            await store.forget(archive.id);
            return { text: `Archive ${archive.id} is no longer held by the server; only its text is left. `
                         + "Use memory_search on it.", archive };
        }
        const a = fork.answers[0];
        let text = a?.content?.trim() || "(no answer)";
        if (a?.finish_reason === "length") text += "\n[the answer was cut off at its token limit]";
        return { text, archive, cached: fork.usage?.prompt_tokens_details?.cached_tokens };
    }
}

/* Hits as the tool hands them to the model: newest context first within equal relevance, each with
 * where it came from, within a budget the model's context can take. */
export function formatHits(hits: Hit[], budget = 8000): string {
    if (hits.length === 0) return "No matches in this session's compacted history.";
    const out: string[] = [];
    let used = 0;
    for (const h of hits) {
        const c = h.chunk;
        const when = c.timestamp ? new Date(c.timestamp).toISOString().replace("T", " ").slice(0, 16) : "?";
        const head = `[${c.kind} · entry ${c.entryId} · ${when}${c.archiveId ? ` · archive ${c.archiveId}` : ""}]`;
        const room = budget - used - head.length - 2;
        if (room < 200) break;
        const body = c.text.length > room ? `${c.text.slice(0, room)} [...]` : c.text;
        out.push(`${head}\n${body}`);
        used += head.length + body.length + 2;
    }
    return out.join("\n\n");
}
