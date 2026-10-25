import assert from "node:assert/strict";
import { test } from "node:test";

import type { SessionEntry } from "@earendil-works/pi-coding-agent";

import { chunksOf } from "../src/chunks.ts";

function message(id: string, message: Record<string, unknown>): SessionEntry {
    return { type: "message", id, parentId: null, timestamp: "2026-10-01T10:00:00.000Z", message } as unknown as SessionEntry;
}

test("each role reads as its own kind", () => {
    const [u] = chunksOf(message("u1", { role: "user", content: "fix the build", timestamp: 5 }));
    assert.deepEqual(u, { id: "u1", entryId: "u1", kind: "user", text: "fix the build", timestamp: 5 });

    const [a] = chunksOf(message("a1", {
        role: "assistant",
        content: [
            { type: "thinking", thinking: "the linker misses memory.cpp" },
            { type: "text", text: "Reconfiguring." },
            { type: "toolCall", id: "t", name: "bash", arguments: { command: "cmake -S . -B build" } },
        ],
    }));
    assert.equal(a.kind, "assistant");
    assert.match(a.text, /^\(reasoning\) the linker misses memory\.cpp\nReconfiguring\.\n-> bash\(\{"command":"cmake/);
    assert.equal(a.timestamp, Date.parse("2026-10-01T10:00:00.000Z"));

    const t = chunksOf(message("t1", { role: "toolResult", toolName: "bash", isError: true,
                                       content: [{ type: "text", text: "x".repeat(5000) }] }));
    assert.ok(t.every((c) => c.kind === "tool"));
    assert.match(t[0].text, /^\[bash, failed\] x+$/);
    assert.match(t[t.length - 1].text, /x\s\[\.\.\. 1000 more characters\]$/);
});

test("a long message is cut into overlapping windows", () => {
    const text = Array.from({ length: 400 }, (_, i) => `word${i}`).join(" ");
    const c = chunksOf(message("u2", { role: "user", content: text }));
    assert.ok(c.length > 1);
    assert.deepEqual(c.map((x) => x.id), c.map((_, i) => `u2#${i}`));
    assert.ok(c.every((x) => x.entryId === "u2" && x.text.length <= 1600));
    assert.equal(c[0].text.slice(-200), c[1].text.slice(0, 200));
    assert.ok(c[c.length - 1].text.endsWith("word399"));
});

test("summaries are kept and bookkeeping entries are not", () => {
    const s = chunksOf({ type: "compaction", id: "c1", parentId: null, timestamp: "2026-10-01T10:00:00.000Z",
                         summary: "## Goal\nship it", firstKeptEntryId: "x", tokensBefore: 9 } as unknown as SessionEntry);
    assert.equal(s[0].kind, "summary");
    assert.deepEqual(chunksOf({ type: "model_change", id: "m", parentId: null, timestamp: "",
                                provider: "p", modelId: "m" } as unknown as SessionEntry), []);
    assert.deepEqual(chunksOf(message("e", { role: "user", content: "   " })), []);
});
