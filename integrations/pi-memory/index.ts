/* radiance-pi-memory -- persistent memory for Pi on a Radiance server.
 *
 * Compaction forks the cached conversation for its summary instead of re-reading it as text, and
 * archives it: Radiance pins the conversation's KV, so memory_ask can later question the whole of
 * it at the cost of the question alone. What compaction removes is also indexed on disk for
 * memory_search, which works on any provider. See README.md. */
import { Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

import { configFromEnv, formatHits, MemoryRuntime } from "./src/memory.ts";

export default function radianceMemory(pi: ExtensionAPI): void {
    const rt = new MemoryRuntime(configFromEnv());

    pi.on("session_start", async (_event, ctx) => {
        await rt.storeOf(ctx);
    });

    pi.on("before_provider_request", (event, ctx) => rt.tag(event.payload, ctx));

    pi.on("session_before_compact", (event, ctx) => rt.compact(event, ctx));

    pi.on("session_compact", async (event, ctx) => {
        try {
            await rt.indexCompaction(event, ctx);
        } catch (e) {
            ctx.ui.notify(`radiance memory: indexing failed (${(e as Error).message})`, "warning");
        }
    });

    pi.registerTool({
        name: "memory_search",
        label: "Memory search",
        description:
            "Search this session's compacted history: messages, tool output, summaries and decision "
            + "records that compaction removed from the context. Returns the best matching excerpts with "
            + "the entry and archive each came from.",
        promptSnippet: "memory_search: find text from earlier in this session that compaction removed",
        promptGuidelines: [
            "When a question concerns work from before the last compaction and the summary lacks the detail, "
            + "use memory_search before guessing or redoing the work.",
        ],
        parameters: Type.Object({
            query: Type.String({ description: "What to look for: names, paths, error text, or a description" }),
            limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 50, description: "Excerpts to return (8)" })),
        }),
        async execute(_id, params, signal, _onUpdate, ctx) {
            const hits = await rt.search(params.query, params.limit ?? 8, ctx, signal);
            return {
                content: [{ type: "text", text: formatHits(hits) }],
                details: { hits: hits.length, archives: [...new Set(hits.map((h) => h.chunk.archiveId))] },
            };
        },
    });

    pi.registerTool({
        name: "memory_ask",
        label: "Memory ask",
        description:
            "Ask a question of an archived part of this session: the model answers with that whole "
            + "conversation in view, as it was before compaction, tool output untruncated. Slower than "
            + "memory_search, and only for sessions on a Radiance server; use it when excerpts are not "
            + "enough, as for why something was decided or what was tried before.",
        promptSnippet: "memory_ask: question the full conversation as it was before a compaction",
        parameters: Type.Object({
            question: Type.String({ description: "The question, specific and self-contained" }),
            archive_id: Type.Optional(Type.String({
                description: "An archive id from memory_search; else the archive the question best matches",
            })),
        }),
        async execute(_id, params, signal, _onUpdate, ctx) {
            const r = await rt.ask(params.question, params.archive_id, ctx, signal);
            const where = r.archive ? `[archive ${r.archive.id}]\n` : "";
            return {
                content: [{ type: "text", text: where + r.text }],
                details: { archive: r.archive?.id, cachedTokens: r.cached },
            };
        },
    });

    pi.registerCommand("memory", {
        description: "This session's archives; /memory drop <id> releases one on the server",
        handler: async (args, ctx) => {
            const [verb, id] = args.trim().split(/\s+/);
            if (verb === "drop" && id) {
                const server = await rt.server(ctx);
                if (!server) {
                    ctx.ui.notify("radiance memory: no Radiance server with /v1/memory", "warning");
                    return;
                }
                try {
                    await server.remove(id);
                    ctx.ui.notify(`radiance memory: released ${id}`, "info");
                } catch (e) {
                    ctx.ui.notify(`radiance memory: ${(e as Error).message}`, "warning");
                }
                return;
            }
            ctx.ui.notify(await rt.describe(ctx), "info");
        },
    });
}
