# Agent Conventions

Workflow established 2026-09-19 for multi-agent sessions in this repo.

## Delegation
- Main agent delegates all coding and testing to subagents.
- Subagents get self-contained context (no reliance on conversation history).

## Question Handling
- Questions/uncertainties are noted and batched for periodic review sessions.
- Don't block work to confer — surface after, then review together.

## Boundaries
- All work confined to this repo folder and subfolders.
- Temp files go in `.gitignore`.

## Review
- Subagent outputs are reviewed against prior decisions before inclusion.
- If a subagent output contradicts past decisions, surface it — don't silently override.

## Notes
- Important observations captured as tests or descriptive "pseudotests."
