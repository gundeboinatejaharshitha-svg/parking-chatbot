# Working rules for this repository

## Environment
- Windows PowerShell. Python venv at .\.venv\Scripts\python. Run modules as `.\.venv\Scripts\python -m ...`.
- This folder is the git repo root. Never run git commands in a parent folder.

## Git workflow (do this automatically, without asking me)
After finishing each logical unit of work (a part, a module, a fix):
1. Run the relevant check (the verification command, or `.\.venv\Scripts\python -m pytest -q` once tests exist). Commit only if it passes. If it fails, fix it or report it; do not commit broken code.
2. Run `git status` and confirm no .env, .venv, parking.db, chroma_db, or *.log files are staged. If any are, unstage them.
3. Stage only files you changed: `git add <paths>`. Never `git add -A` or `git add .` unless `git status` shows only intended files.
4. Commit with a Conventional Commit message (feat:, fix:, test:, docs:, chore:, refactor:). One commit per logical unit, subject under 72 characters.
5. Push with `git push` after each commit if a remote named origin exists and works. If push fails (auth or network), say so and continue; do not retry in a loop.
6. Never force-push, rewrite history, or change the remote URL without asking me.

## Code rules
- Do not edit app/data/sql_store.py or app/data/seed.py unless I ask.
- Type hints and short docstrings everywhere. Use logging, never print (scripts/ may print).
- Never log full car plates, names, or booking references; use the mask_* helpers.
- Reservation validation happens in code, never in the LLM. The LLM never accesses the reservations table.
- Config only via app/config.py and environment variables. No hardcoded secrets.

## After every task
- Update the Progress checklist in docs/PROJECT_CONTEXT.md when a stage item is completed (include that file in the same commit).
- End with a 3-line summary: what changed, what was verified, the commit hash.

## Step 2: Let Copilot run commands without clicking each one

By default VS Code asks you to approve every terminal command. In VS Code, open Settings, search for `chat.tools.terminal.autoApprove`, and add an allow-list so only safe commands run unattended:

```json
"chat.tools.terminal.autoApprove": {
  "git status": true,
  "git add": true,
  "git commit": true,
  "git push": true,
  "git log": true,
  "git diff": true,
  "git remote -v": true,
  "/^\\.\\\\\\.venv\\\\Scripts\\\\python\\b/": true,
  "git push --force": false,
  "git reset": false,
  "Remove-Item": false
}
```

Setting names change between VS Code versions, so if that key isn't found, search for "auto approve" in settings. Avoid the global "approve all tools" option. With an allow-list, a risky command such as a force-push or a file deletion still stops and asks you.

## Step 3: The GitHub remote still needs you once

Copilot can't create a GitHub account's repo or sign in for you. Do this one time:

1. Create the empty repo `parking-chatbot` on GitHub (no README, no `.gitignore`).
2. Tell Copilot: `The repo is created at https://github.com/gundeboinatejaharshitha-svg/parking-chatbot.git. Run git remote remove origin, then git remote add origin <that url>, git remote -v, and git push -u origin main.`
3. Approve the browser sign-in that opens. After this, every push is automatic.

If you have the GitHub CLI installed (`gh --version`), Copilot can create the repo itself with `gh repo create parking-chatbot --public --source=. --remote=origin --push`.

## From now on

Each stage prompt just needs one extra line at the end: *"Follow .github/copilot-instructions.md for verification, commits and push."*

## Two limits

- **Copilot can pause or skip steps** in long sessions. Run `git log --oneline` now and then, and ask it to commit if something is missing.
- **Check the first few commits yourself.** Look at `git show --stat` to confirm no `.env` or database file got in. Once those look right, you can trust the pattern.

When the instructions file is committed, paste the Part C prompt from earlier into Copilot Agent mode with the extra line above.