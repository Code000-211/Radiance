/* chunks.ts -- what of a Pi session is searched, in pieces a search can hand back.
 *
 * One chunk per message, cut into windows when it is long. Tool output is the bulk of a coding
 * agent's context and mostly reproducible (a file can be read again, a command run again), so it
 * is kept shorter and ranked lower (store.ts). An assistant turn keeps its reasoning: "why did we
 * decide X" is answered there more often than in the reply. */
import type { SessionEntry } from "@earendil-works/pi-coding-agent";

export type ChunkKind = "user" | "assistant" | "tool" | "summary" | "decision";

export interface Chunk {
    id: string;              /* the entry id, with "#n" for the n-th window of a long one */
    entryId: string;
    kind: ChunkKind;
    text: string;
    timestamp: number;       /* ms since the epoch */
    archiveId?: string;      /* the compaction that removed it from the live context */
}

const WINDOW = 1600;         /* characters a chunk holds */
const OVERLAP = 200;         /* carried into the next window, so a sentence cut at the edge is
                              * still found whole in one of them */
const TOOL_CHARS = 4000;     /* of a tool's output; Pi's own summariser keeps 2000 */
const ARGS_CHARS = 400;      /* of a tool call's arguments */

function windows(text: string): string[] {
    if (text.length <= WINDOW) return [text];
    const out: string[] = [];
    for (let at = 0; at < text.length; at += WINDOW - OVERLAP) {
        out.push(text.slice(at, at + WINDOW));
        if (at + WINDOW >= text.length) break;
    }
    return out;
}

function clip(text: string, n: number): string {
    return text.length <= n ? text : `${text.slice(0, n)} [... ${text.length - n} more characters]`;
}

/* The text parts of a message's content, which is a string or an array of typed parts. */
function textOf(content: unknown): string {
    if (typeof content === "string") return content;
    if (!Array.isArray(content)) return "";
    const parts: string[] = [];
    for (const p of content) {
        if (p && typeof p === "object" && (p as { type?: unknown }).type === "text") {
            const t = (p as { text?: unknown }).text;
            if (typeof t === "string") parts.push(t);
        }
    }
    return parts.join("\n");
}

function assistantText(content: unknown): string {
    if (!Array.isArray(content)) return textOf(content);
    const parts: string[] = [];
    for (const p of content as Array<Record<string, unknown>>) {
        if (!p || typeof p !== "object") continue;
        if (p.type === "thinking" && typeof p.thinking === "string" && p.thinking.trim())
            parts.push(`(reasoning) ${p.thinking.trim()}`);
        else if (p.type === "text" && typeof p.text === "string") parts.push(p.text);
        else if (p.type === "toolCall" && typeof p.name === "string")
            parts.push(`-> ${p.name}(${clip(JSON.stringify(p.arguments ?? {}), ARGS_CHARS)})`);
    }
    return parts.join("\n");
}

function stamp(entry: SessionEntry, message?: { timestamp?: unknown }): number {
    if (message && typeof message.timestamp === "number") return message.timestamp;
    const t = Date.parse(entry.timestamp);
    return Number.isFinite(t) ? t : 0;
}

/* The chunks of one session entry; none for entries that say nothing searchable (model changes,
 * labels, custom state). */
export function chunksOf(entry: SessionEntry): Chunk[] {
    let kind: ChunkKind;
    let text = "";
    let ts = stamp(entry);
    if (entry.type === "message") {
        const m = entry.message as unknown as { role?: string; content?: unknown; toolName?: unknown;
                                                isError?: unknown; timestamp?: unknown };
        ts = stamp(entry, m);
        if (m.role === "user") {
            kind = "user";
            text = textOf(m.content);
        } else if (m.role === "assistant") {
            kind = "assistant";
            text = assistantText(m.content);
        } else if (m.role === "toolResult") {
            kind = "tool";
            const name = typeof m.toolName === "string" ? m.toolName : "tool";
            text = `[${name}${m.isError === true ? ", failed" : ""}] ${clip(textOf(m.content), TOOL_CHARS)}`;
        } else {
            return [];
        }
    } else if (entry.type === "compaction" || entry.type === "branch_summary") {
        kind = "summary";
        text = entry.summary;
    } else {
        return [];
    }
    text = text.trim();
    if (!text) return [];
    const pieces = windows(text);
    return pieces.map((t, i) => ({
        id: pieces.length === 1 ? entry.id : `${entry.id}#${i}`,
        entryId: entry.id,
        kind,
        text: t,
        timestamp: ts,
    }));
}
