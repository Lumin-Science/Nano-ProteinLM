<div class="ai">

# Reviewing AI-authored documentation

</div>

<div class="ai">

Each agent-authored or edited paragraph in `README.md` and `docs/` gets its own `ai` wrapper until the owner reviews it; VS Code’s Markdown preview shows the paragraph in blue with a blue left border, and each wrapper can be removed independently.

</div>

<div class="ai">

## View the blue text

</div>

<div class="ai">

Open this repository’s folder in VS Code, then open a Markdown file and choose **Markdown: Open Preview to the Side** from the Command Palette (`Cmd+Shift+P` on macOS, `Ctrl+Shift+P` on Windows/Linux); VS Code loads the tracked `.vscode/ai-review.css` through the workspace’s `markdown.styles` setting. Workspace styles require a trusted folder, so Restricted Mode can prevent them from loading.

</div>

<div class="ai">

The default is blue (`#0969da`) on light themes and a lighter blue (`#58a6ff`) on dark themes. The selectors also cover paragraphs, list items, table cells, headings, links and code so their default styles do not hide the review color. The source editor keeps its normal syntax colors; GitHub does not load this local stylesheet. See [VS Code’s custom Markdown CSS documentation](https://code.visualstudio.com/docs/languages/markdown#_using-your-own-css).

</div>

<div class="ai">

## Mark one paragraph at a time

</div>

<div class="ai">

Use a separate wrapper for each changed paragraph, with blank lines around the Markdown inside so formatting is preserved; never wrap multiple paragraphs, a whole section or a whole document together.

</div>

<div class="ai">

```markdown
<div class="ai">

This paragraph awaits owner review.

</div>

<div class="ai">

This second paragraph can be reviewed independently.

</div>
```

</div>

<div class="ai">

Give a changed heading, table or fenced code block its own wrapper, separate from surrounding paragraphs; keep the HTML outside fenced code so commands stay copyable. For a changed phrase, list item or table cell, use a span without splitting the surrounding paragraph, list or table.

</div>

<div class="ai">

```markdown
Existing text with <span class="ai">an updated phrase</span>.

- <span class="ai">An updated list item.</span>
```

</div>

<div class="ai">

## Leave an inline fix request

</div>

<div class="ai">

Type `aitofix` and press **Tab** in a Markdown file, then replace the selected placeholder with your request. The shortcut inserts `<aitofix>your note</aitofix>`, which works in the standard VS Code preview without an extension. Literal `\aitofix{...}` is not a Markdown command; use the snippet or type the short HTML tag directly.

</div>

<div class="ai">

Place one note beside the paragraph it refers to so the request and its eventual resolution can be reviewed independently. Open notes are red and labeled **AI TO FIX**; adding the `resolved` attribute makes the note orange and labels it **AI FIXED**, including inside a blue `ai` paragraph. The labels are added by the preview stylesheet; the original note remains in the Markdown source.

</div>

<div class="ai">

```markdown
The benchmark reports precision over all evaluation chains. <aitofix>Explain how the chains are selected.</aitofix>

The benchmark reports precision over all evaluation chains. <aitofix resolved>Explain how the chains are selected. — Fixed: added the selection rule and checked it against the evaluator.</aitofix>
```

</div>

<div class="ai">

Ask the agent to address the unresolved `aitofix` notes in the file. It reads the nearby text, implements and checks each fix, then adds `resolved` while preserving your request and a short explanation of the result; unresolved or blocked work stays red. Code-block examples are not requests. After reviewing a completed fix, remove its note; remove only `resolved` to reopen it.

</div>

<div class="ai">

The optional `aitofixdone` snippet inserts a resolved note with a `Fixed:` placeholder. If Tab does not expand a prefix, use **Insert Snippet** from the Command Palette and choose **AI fix request**; the tracked workspace settings enable Markdown snippet completion. These notes do not launch an agent automatically.

</div>

<div class="ai">

## Finish the review

</div>

<div class="ai">

After reviewing a paragraph, the owner removes just its surrounding `div` or `span` tags while keeping the content; agents leave the wrappers in place until the owner directs otherwise.

</div>

<div class="ai">

When applying this setup to another repository, add `.vscode/ai-review.css` to the existing `markdown.styles` array in `.vscode/settings.json`, preserving other settings and styles; also copy `.vscode/ai-review.code-snippets` and enable `editor.tabCompletion` for Markdown to use the fix-note shortcuts.

</div>

<div class="ai">

If `.vscode/` is ignored, reopen the directory with `!/.vscode/`, ignore other entries with `/.vscode/*`, and allow `!/.vscode/settings.json`, `!/.vscode/ai-review.css` and `!/.vscode/ai-review.code-snippets`; Git cannot re-include files while their parent directory remains ignored. Confirm the files are not ignored with `git check-ignore .vscode/settings.json .vscode/ai-review.css .vscode/ai-review.code-snippets`, then track them with Git.

</div>
