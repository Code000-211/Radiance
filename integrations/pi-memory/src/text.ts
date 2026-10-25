/* text.ts -- lexical search over what a conversation said.
 *
 * An agent's history is full of identifiers: paths, symbols, flags, error codes. Embeddings blur
 * them, so the lexical half of the search keeps each one whole and also indexes its parts, so that
 * `core/mem/prefix.cpp` answers a search for `prefix`, and `snapshotOwed` one for `owed`. */

/* The words of `text`, lower-cased: every identifier-like run whole, then its parts when it has
 * more than one. Runs of one character carry nothing a search could want. */
export function terms(text: string): string[] {
    const out: string[] = [];
    for (const run of text.match(/[A-Za-z0-9_./:#-]+/g) ?? []) {
        const word = run.replace(/^[./:#-]+|[./:#-]+$/g, "");
        if (word.length < 2) continue;
        out.push(word.toLowerCase());
        const parts = word
            .split(/[_./:#-]+/)
            .flatMap((p) => p.split(/(?<=[a-z0-9])(?=[A-Z])/))
            .filter((p) => p.length >= 2)
            .map((p) => p.toLowerCase());
        if (parts.length > 1) out.push(...parts);
    }
    return out;
}

export interface Scored {
    id: string;
    score: number;
}

/* Okapi BM25 over documents added one at a time; nothing is ever removed, as a conversation's
 * history only grows. k1 and b are the usual values. */
export class Bm25 {
    private readonly k1 = 1.2;
    private readonly b = 0.75;
    private readonly tf = new Map<string, Map<string, number>>();
    private readonly len = new Map<string, number>();
    private readonly df = new Map<string, number>();
    private total = 0;

    get size(): number {
        return this.len.size;
    }

    add(id: string, text: string): void {
        if (this.len.has(id)) return;
        const counts = new Map<string, number>();
        const words = terms(text);
        for (const w of words) counts.set(w, (counts.get(w) ?? 0) + 1);
        for (const w of counts.keys()) this.df.set(w, (this.df.get(w) ?? 0) + 1);
        this.tf.set(id, counts);
        this.len.set(id, words.length);
        this.total += words.length;
    }

    /* The best `limit` documents for `query`, best first; documents sharing no word with it are
     * left out. */
    search(query: string, limit: number): Scored[] {
        const n = this.len.size;
        if (n === 0) return [];
        const avg = this.total / n || 1;
        const q = [...new Set(terms(query))];
        const scores = new Map<string, number>();
        for (const w of q) {
            const df = this.df.get(w);
            if (!df) continue;
            const idf = Math.log(1 + (n - df + 0.5) / (df + 0.5));
            for (const [id, counts] of this.tf) {
                const f = counts.get(w);
                if (!f) continue;
                const dl = this.len.get(id) ?? 0;
                const s = (idf * f * (this.k1 + 1)) / (f + this.k1 * (1 - this.b + (this.b * dl) / avg));
                scores.set(id, (scores.get(id) ?? 0) + s);
            }
        }
        return [...scores.entries()]
            .map(([id, score]) => ({ id, score }))
            .sort((a, b) => b.score - a.score || (a.id < b.id ? -1 : 1))
            .slice(0, limit);
    }
}

/* Reciprocal rank fusion: each list votes 1 / (k + rank) for its members. Lists rank by
 * incomparable scores (BM25 against cosine), so only the ranks are combined. */
export function fuse(lists: Scored[][], k = 60): Map<string, number> {
    const out = new Map<string, number>();
    for (const list of lists)
        list.forEach((s, rank) => out.set(s.id, (out.get(s.id) ?? 0) + 1 / (k + rank + 1)));
    return out;
}

export function cosine(a: readonly number[], b: readonly number[]): number {
    let dot = 0;
    let na = 0;
    let nb = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return na > 0 && nb > 0 ? dot / Math.sqrt(na * nb) : 0;
}
