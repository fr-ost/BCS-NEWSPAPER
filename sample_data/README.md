# Sample data (offline mode)

Everything here is **fictional** and exists only so `python -m app run --offline`
and the tests can exercise the full pipeline without network access or model
quota. Titles are prefixed with `[SAMPLE]` / `[নমুনা]` and URLs point to
`example.com`. Offline output is never sent to Telegram.

- `feeds/<source_id>.xml` — RSS read in place of the live feed for that source id.
- `articles/<last-url-path-segment>.html` — article pages for offline extraction.
