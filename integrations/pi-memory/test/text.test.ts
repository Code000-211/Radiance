import assert from "node:assert/strict";
import { test } from "node:test";

import { Bm25, cosine, fuse, terms } from "../src/text.ts";

test("an identifier is a term whole and by its parts", () => {
    const t = terms("see core/mem/prefix.cpp and snapshotOwed()");
    assert.ok(t.includes("core/mem/prefix.cpp"));
    assert.ok(t.includes("prefix"));
    assert.ok(t.includes("cpp"));
    assert.ok(t.includes("snapshotowed"));
    assert.ok(t.includes("owed"));
    assert.ok(!t.includes("a"));
});

test("bm25 ranks the document that says the rare word most", () => {
    const b = new Bm25();
    b.add("1", "the build failed on the linker");
    b.add("2", "the tests passed");
    b.add("3", "the linker linker error was a missing source file");
    b.add("3", "added twice is added once");
    assert.equal(b.size, 3);
    const r = b.search("linker", 10);
    assert.deepEqual(r.map((s) => s.id), ["3", "1"]);
    assert.deepEqual(b.search("nothing here", 10), []);
});

test("rank fusion favours what both lists rank high", () => {
    const f = fuse([[{ id: "a", score: 9 }, { id: "b", score: 1 }], [{ id: "b", score: 0.9 }, { id: "a", score: 0.8 }],
                    [{ id: "b", score: 0.5 }]]);
    assert.ok((f.get("b") ?? 0) > (f.get("a") ?? 0));
});

test("cosine", () => {
    assert.equal(cosine([1, 0], [1, 0]), 1);
    assert.equal(cosine([1, 0], [0, 1]), 0);
    assert.equal(cosine([0, 0], [1, 1]), 0);
});
