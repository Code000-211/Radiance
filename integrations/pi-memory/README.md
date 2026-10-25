# radiance-pi-memory

Long-term memory for the [Pi](https://github.com/earendil-works/pi) coding agent, built on the
Radiance server's conversation memory (`/v1/memory`, docs/GUIDE.md §5.17).

When Pi compacts, it normally serialises the old turns as text, truncates the tool output and asks
the model for a summary in a fresh context. With this extension, on a model Radiance serves:

- **The summary is a fork of the cached conversation.** Radiance still holds the conversation's KV,
  so the summary question is appended to it as a new turn and only that turn is prefilled. The model
  writes the summary with the whole conversation in view: tool output untruncated, its own reasoning
  included. A second question in the same fork writes decision records (what was decided, why, what
  was rejected, which files) for the search below.
- **The conversation is archived.** Radiance pins its cached blocks and its deepest linear-state
  snapshot, so the conversation can still be questioned after Pi stopped sending it.
- **What compaction removed is indexed** on disk, per session, for search.

The agent gets two tools:

| tool | what it does |
|---|---|
| `memory_search {query, limit?}` | BM25 over the compacted messages, tool output, summaries and decision records, fused with embedding similarity when an embeddings endpoint is configured; ranks decision records and user turns above tool output, and lifts the chunks of the compaction whose summary matches |
| `memory_ask {question, archive_id?}` | asks the archived conversation itself, through a fork; answers from the whole of it at the cost of the question. Without an `archive_id` it picks the archive the question's best search hits came from |

and the command `/memory`, which lists this session's compactions and what the server still holds of
them; `/memory drop ID` releases one on the server.

Everything Radiance does here can fail without harm. If the server is unreachable, keeps no memory
for this client, or the fork does not produce a complete summary, Pi compacts as it always does.
The text index is built either way, so `memory_search` works on any provider. Pi's session file
stays the record of what was said; an archive is a fast way back into it.

## Install

```sh
pi install /path/to/Radiance/integrations/pi-memory
# or, for one run
pi -e /path/to/Radiance/integrations/pi-memory/index.ts
```

Serve the model with a prefix cache (the default) and, for long sessions, a host tier:

```sh
radiance --model m.rad --prefix-cache-mib 32768 --prefix-cache-ttl 7200 --memory-archives 8
```

Point Pi at it as an OpenAI-compatible (or Anthropic) provider. The extension tags every request to
that server with Pi's session id (`session_id`, or `metadata.user_id` on the Anthropic API), which
is the name Radiance keeps the conversation under; requests to other providers are left alone.

Without `--api-key`, Radiance answers the memory routes only on loopback. A remote server needs a
key, which the extension takes from the model's own provider configuration or from
`RADIANCE_MEMORY_KEY`.

## Configuration

| environment | default | |
|---|---|---|
| `RADIANCE_MEMORY_URL` | the model's base URL | the Radiance server's root (`http://host:8000`) |
| `RADIANCE_MEMORY_KEY` | the model's API key | bearer key for the memory routes |
| `RADIANCE_MEMORY_DIR` | `~/.pi/agent/radiance-memory` | where the per-session index is kept |
| `RADIANCE_MEMORY_EMBED_URL`, `_EMBED_MODEL`, `_EMBED_KEY` | unset | an OpenAI-compatible `/v1/embeddings` endpoint (a second Radiance process serving an embedding model, for example); unset, search is lexical only |
| `RADIANCE_MEMORY_SUMMARY_THINKING` | 1024 | reasoning tokens the summary may spend |
| `RADIANCE_MEMORY_SUMMARY_MAX_TOKENS` | 8192 | the summary's token limit; a summary cut off by it is discarded and Pi compacts instead |
| `RADIANCE_MEMORY_ASK_THINKING`, `_ASK_MAX_TOKENS` | 512, 2048 | the same for `memory_ask` |
| `RADIANCE_MEMORY_DECISIONS` | on | `0` skips the decision records |

The server keeps `--memory-archives` archives across all its clients. When it is full, the extension
releases the oldest archive it wrote itself (ids starting `pi-`), never another client's; the
released compaction stays searchable as text.

## On disk

`$RADIANCE_MEMORY_DIR/<session id>/` holds three append-only JSON-lines files: `chunks.jsonl` (the
indexed text), `vectors.jsonl` (their embeddings, when configured) and `archives.jsonl` (one record
per compaction: the summary, the Radiance archive id and whether it is still held). A torn last line
from a crash is skipped on load.

## Development

```sh
npm install
npm run check     # tsc --noEmit
npm test          # node --test: the search, the store, and the runtime against a fake server
```

The runtime tests run against a stand-in for the memory routes. The routes themselves are tested in
Radiance's `tests/server_test.cpp` and the pinning in `tests/mem_test.cpp`.
