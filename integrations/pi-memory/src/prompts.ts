/* prompts.ts -- what the fork asks the model at compaction and on recall.
 *
 * A fork appends its question as a new user turn after the conversation itself (memory.cpp), so
 * the model answers with the whole conversation in view, tool output untruncated and its own
 * reasoning included. The risk Pi avoids by serialising the transcript as text instead, a model
 * that carries on with the task, is met by saying plainly that the task is paused and that no
 * tool may be called. */
import type { FileOperations } from "@earendil-works/pi-coding-agent";

/* Pi's summary format (docs/compaction.md), so a summary the fork wrote reads like one Pi wrote. */
export function summaryQuestion(keepTokens: number, custom?: string): string {
    const focus = custom?.trim() ? `\n\nFocus especially on: ${custom.trim()}` : "";
    return `Pause the task. Do not call any tool and do not continue the work.

The conversation above is about to be compacted: everything except roughly the last ${keepTokens} tokens will be replaced by the summary you write now, and the work will continue from that summary alone. Write it in exactly this format:

## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
- [Requirements the user stated]

## Progress
### Done
- [x] [Completed tasks]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues, if any]

## Key Decisions
- **[Decision]**: [Rationale]

## Next Steps
1. [What should happen next]

## Critical Context
- [Data needed to continue: exact paths, names, commands, numbers, error messages]

Be specific and complete; anything left out is gone from the working context.${focus}`;
}

/* Decision records: what a later question about the past most often wants, written by the model
 * that made the decisions, with the reasons still in its context. One line each, so each becomes
 * a search entry of its own. */
export const DECISIONS_QUESTION = `Pause the task. Do not call any tool and do not continue the work.

List every decision made in the conversation above that someone might later ask about: a choice between approaches, a design or API decision, a fix and its cause, something tried and abandoned. One line per decision, each starting with "- ", in the form:

- DECISION: <what was decided> | WHY: <the reason> | REJECTED: <the alternatives, or none> | FILES: <paths, or none>

Write nothing else.`;

/* A recall question, asked of an archived conversation. */
export function recallQuestion(question: string): string {
    return `Pause the task. Do not call any tool and do not continue the work.

The conversation above has since been compacted away. Answer this question about it from what the conversation shows, briefly and specifically (exact names, paths, numbers, and the reasons given). If the conversation does not show the answer, say so.

Question: ${question}`;
}

/* The lines of a decision list that are records. */
export function decisionLines(text: string): string[] {
    return text
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => l.startsWith("- ") && l.length > 4)
        .map((l) => l.slice(2).trim());
}

/* Pi's file lists, appended to a summary as its default compaction does: files only read, and
 * files written or edited. */
export function fileLists(ops: FileOperations | undefined): string {
    if (!ops) return "";
    const modified = [...new Set([...ops.written, ...ops.edited])].sort();
    const read = [...ops.read].filter((f) => !ops.written.has(f) && !ops.edited.has(f)).sort();
    let out = "";
    if (read.length) out += `\n\n<read-files>\n${read.join("\n")}\n</read-files>`;
    if (modified.length) out += `\n\n<modified-files>\n${modified.join("\n")}\n</modified-files>`;
    return out;
}
