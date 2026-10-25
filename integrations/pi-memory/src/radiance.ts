/* radiance.ts -- the Radiance server's /v1/memory routes (core/server/memory.cpp), and an
 * OpenAI-compatible embeddings endpoint for the search's semantic half. */

export type ForkSource = { session: string } | { archive: string };

export interface ForkAnswer {
    index: number;
    question: string;
    content: string;
    reasoning_content?: string;
    finish_reason: string;
}

export interface ForkResult {
    answers: ForkAnswer[];
    source_tokens: number;
    warning?: string;
    usage?: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
        prompt_tokens_details?: { cached_tokens: number };
    };
}

export interface ArchiveStat {
    id: string;
    session_id: string;
    tokens: number;
    pinned_tokens: number;
    snapshot_tokens: number;
    age_s: number;
    on_disk?: boolean;      /* written to the server's --memory-dir: survives its restart */
}

export interface ForkOptions {
    /* Write the snapshot an archive resumes from at the conversation's end (compaction does). */
    checkpointAtEnd?: boolean;
    maxTokens?: number;
    /* Reasoning tokens before the block is closed; recall answers rarely need many. */
    thinkingBudget?: number;
    signal?: AbortSignal;
}

/* A failed call, with the server's own sentence and its status. */
export class RadianceError extends Error {
    readonly status: number;
    constructor(status: number, message: string) {
        super(message);
        this.status = status;
    }
}

async function failure(res: Response): Promise<RadianceError> {
    let why = `${res.status} ${res.statusText}`;
    try {
        const j = (await res.json()) as { error?: { message?: string } };
        if (j?.error?.message) why = `${res.status}: ${j.error.message}`;
    } catch {
        /* Not JSON; the status says enough. */
    }
    return new RadianceError(res.status, why);
}

export class RadianceMemory {
    readonly base: string;
    private readonly key: string | undefined;

    /* `base` is the server's root, as http://host:8000, not its /v1. */
    constructor(base: string, key?: string) {
        this.base = base.replace(/\/+$/, "");
        this.key = key;
    }

    private headers(): Record<string, string> {
        const h: Record<string, string> = { "Content-Type": "application/json" };
        if (this.key) h.Authorization = `Bearer ${this.key}`;
        return h;
    }

    private async call<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
        const res = await fetch(this.base + path, {
            method,
            headers: this.headers(),
            body: body === undefined ? undefined : JSON.stringify(body),
            signal,
        });
        if (!res.ok) throw await failure(res);
        return (await res.json()) as T;
    }

    /* Whether the server keeps memory and lets this client use it: 200 on the archive list. */
    async available(signal?: AbortSignal): Promise<boolean> {
        try {
            const res = await fetch(`${this.base}/v1/memory/archive`, { headers: this.headers(), signal });
            return res.ok;
        } catch {
            return false;
        }
    }

    fork(source: ForkSource, questions: string[], o: ForkOptions = {}): Promise<ForkResult> {
        const body: Record<string, unknown> = { source, questions };
        if (o.checkpointAtEnd) body.checkpoint_at_end = true;
        if (o.maxTokens) body.max_tokens = o.maxTokens;
        if (o.thinkingBudget !== undefined) body.thinking_token_budget = o.thinkingBudget;
        return this.call<ForkResult>("POST", "/v1/memory/fork", body, o.signal);
    }

    archive(sessionId: string, archiveId: string, signal?: AbortSignal): Promise<ArchiveStat> {
        return this.call<ArchiveStat>("POST", "/v1/memory/archive",
                                      { session_id: sessionId, archive_id: archiveId }, signal);
    }

    async archives(signal?: AbortSignal): Promise<ArchiveStat[]> {
        return (await this.call<{ data: ArchiveStat[] }>("GET", "/v1/memory/archive", undefined, signal)).data;
    }

    remove(id: string, signal?: AbortSignal): Promise<unknown> {
        return this.call("DELETE", `/v1/memory/archive/${encodeURIComponent(id)}`, undefined, signal);
    }
}

/* Embeddings from any OpenAI-compatible endpoint: EmbeddingGemma 2 in a second Radiance process
 * on the CPU host backend, or anything else. */
export class Embedder {
    private readonly url: string;
    private readonly model: string;
    private readonly key: string | undefined;

    constructor(url: string, model: string, key?: string) {
        this.url = url.replace(/\/+$/, "");
        this.model = model;
        this.key = key;
    }

    async embed(texts: string[], signal?: AbortSignal): Promise<number[][]> {
        if (texts.length === 0) return [];
        const headers: Record<string, string> = { "Content-Type": "application/json" };
        if (this.key) headers.Authorization = `Bearer ${this.key}`;
        const res = await fetch(`${this.url}/v1/embeddings`, {
            method: "POST",
            headers,
            body: JSON.stringify({ model: this.model, input: texts }),
            signal,
        });
        if (!res.ok) throw await failure(res);
        const j = (await res.json()) as { data: Array<{ index: number; embedding: number[] }> };
        const out: number[][] = new Array(texts.length);
        for (const d of j.data) out[d.index] = d.embedding;
        return out;
    }
}
