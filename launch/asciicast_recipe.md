# How to record the asciicast (Devrel's nice-to-have launch asset)

I (Claude) can't record an asciicast in this session — that needs an
interactive terminal. Here's the exact recipe for you to run.

## Install asciinema

```bash
brew install asciinema    # macOS
# or
pip install asciinema     # any OS
```

## Record the silent demo (the killer one)

```bash
asciinema rec -t "adaptergate demo silent — silent slice regression in 60s" silent.cast
```

In the new shell that opens, run exactly this:

```bash
pip install --quiet 'adaptergate[demo]==0.5.3'
adaptergate demo silent
```

Wait for it to finish. Then `exit` to stop recording.

## Upload + embed

```bash
asciinema upload silent.cast
```

Asciinema returns a URL like `https://asciinema.org/a/XXXXXX`. Paste
that URL at the very top of `README.md`, right under the project title.
Asciinema auto-renders it as an inline player on github.com.

## Suggested markup

```markdown
# adaptergate

[![asciicast](https://asciinema.org/a/XXXXXX.svg)](https://asciinema.org/a/XXXXXX)

**CI gate for per-tenant LoRA adapters that update online.**
...
```

The `.svg` thumbnail is what HN readers actually click. Make sure the
recording is under 90 seconds — anything longer and click-through dies.

## Optional: trim the install step

If `pip install` is slow on your machine (>15 s), record only the demo
run from a venv where `adaptergate[demo]` is already installed. The
viewer doesn't need to see pip output — they need to see the gate fire.

## After recording

- Commit a link (or the `.cast` file if small) to `launch/asciicast/`
- Update the README's "60-second demo" section to embed the player
- Reference the asciicast in the Show HN body so the URL preview card
  pulls a thumbnail
