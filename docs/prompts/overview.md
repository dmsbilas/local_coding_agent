# Prompts (`prompts/`)

Prompts are **plain strings**. Keep LLM instructions here so agent graphs stay focused on control flow.

## Module map

| File | Agent(s) | Exports |
|------|----------|---------|
| `prompts/basic.py` | `agents.basic` | `CODE_GEN_SYSTEM_PROMPT`, `VALIDATE_SYSTEM_PROMPT` |
| `prompts/react.py` | `react_ollama`, `react_ai_doer` | `CODE_GEN_SYSTEM_PROMPT`, `VALIDATE_SYSTEM_PROMPT`, `PLAN_SYSTEM_PROMPT` |
| `prompts/__init__.py` | — | Re-exports for convenience |

## Prompt roles

### Code generation

Tells the model it is inside an autonomous loop: implement the request, honor `[VALIDATOR]` feedback, emit fenced source, never write to disk itself.

### Validation

Forces structured JSON:

```json
{
  "approved": true,
  "feedback": "",
  "files": [{"relative_path": "app.py", "content": "..."}]
}
```

`approved=true` must include at least one valid file; the agent then writes automatically.

### Planning (`PLAN_SYSTEM_PROMPT`)

Used only when `PLAN_MODE=plan_and_execute`. Returns `{ "steps": ["...", "..."] }`.

## Editing prompts safely

- Prefer changing `prompts/*.py` over hard-coding strings inside agents.
- Mention path rules (output directory, protected names) so the model does not invent `agents/foo.py` paths.
- Keep validator JSON shape stable — parsers in `utils.parsing` and agent nodes expect `approved` / `feedback` / `files`.

## Adding a new prompt pack

1. Create `prompts/my_domain.py`.
2. Export from `prompts/__init__.py`.
3. Import in your agent node.
4. Document here and in the agent’s doc page.
