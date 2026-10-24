# BCS Newspaper Engine v1.0

A personal, cloud-run current-affairs briefing for **Bangladesh Civil Service (BCS)** preparation.

Every morning (target **08:00 Asia/Dhaka**) a GitHub Actions job:

1. collects recent news from verified RSS/Atom feeds,
2. merges duplicate reports of the same event,
3. screens stories for BCS relevance with an AI model (one lightweight call): **OpenAI** by default, with **Gemini** (free tier) as automatic fallback,
4. reads the accessible article text and writes **grounded Bengali study notes**,
5. ranks stories with a configurable rubric (5–8 on a normal day, fewer on a thin day, never padded),
6. writes **three validated MCQs** from verified facts,
7. sends **one organised edition** on Telegram to everyone who subscribed to the bot: you and your friends (split into several messages only when needed), followed by a **PDF copy** of the same edition.

Expected cost: roughly **US$2–8 per month (≈ ৳250–1,000)** of prepaid OpenAI credit, depending on the model. If the credit runs out, the free Gemini fallback takes over, so the briefing keeps coming. GitHub Actions and Telegram are free. Details in [Cost](#cost-and-free-tier-limits).

---

## Contents

- [What a briefing looks like](#what-a-briefing-looks-like)
- [Manual runs and the PDF](#manual-runs-and-the-pdf)
- [Architecture and data flow](#architecture-and-data-flow)
- [Full setup guide](#full-setup-guide)
- [Subscribers: you and your friends](#subscribers-you-and-your-friends)
- [Running locally](#running-locally)
- [Manual runs and modes](#manual-runs-and-modes)
- [News sources](#news-sources)
- [Factual grounding: how hallucinations are kept out](#factual-grounding-how-hallucinations-are-kept-out)
- [Partial editions and quota exhaustion](#partial-editions-and-quota-exhaustion)
- [State persistence and recovery](#state-persistence-and-recovery)
- [Logs and troubleshooting](#logs-and-troubleshooting)
- [Cost and free-tier limits](#cost-and-free-tier-limits)
- [AI providers: OpenAI with Gemini fallback](#ai-providers-openai-with-gemini-fallback)
- [Disabling the schedule](#disabling-the-schedule)
- [Switching to a local model](#switching-to-a-local-model)
- [Testing](#testing)
- [Security and privacy](#security-and-privacy)
- [Configuration reference](#configuration-reference)

---

## What a briefing looks like

```
📰 বিসিএস দৈনিক ব্রিফিং — ২৩ অক্টোবর ২০২৬
📚 ৬টি খবর · ⏱ পড়তে প্রায় ১০ মিনিট

১. <headline in Bengali>
🏷 অর্থনীতি | প্রিলি + লিখিত | স্কোর ৭৪/১০০
📰 Dhaka Tribune, The Daily Star · ২২ অক্টোবর ২০২৬
কী ঘটেছে: …                       ← summary grounded in the article
বিসিএস-এ কেন গুরুত্বপূর্ণ: …
গুরুত্বপূর্ণ তথ্য (প্রতিবেদন থেকে): • …     ← each fact matched to a quote in the article
📌 প্রিলির জন্য মনে রাখুন: • …
উদ্ধৃত দাবি (যাচাইকৃত তথ্য নয়): • X: …
✍️ লিখিত পরীক্ষার দৃষ্টিকোণ (AI বিশ্লেষণ): …  ← clearly labelled analysis
পটভূমি (প্রতিবেদনের বাইরের সাধারণ প্রেক্ষাপট): …
🔗 Dhaka Tribune | The Daily Star
🏛 দাপ্তরিক সূত্রে যাচাই: Bangladesh Bank

… more stories …

🧠 আজকের অনুশীলনী (৩টি MCQ)       ← questions first
━━━━━━━━━━━━
✅ উত্তর ও ব্যাখ্যা (ট্যাপ করে দেখুন)  ← answers hidden behind Telegram spoilers
```

After the text messages, each subscriber also receives `BCS-Briefing-<date>.pdf`: the same edition laid out for A4 (good for printing or saving), with the MCQ answers on the last page.

Run `python -m app run --offline` to render a full sample edition (text and, if Chrome/Chromium is installed, the PDF) from bundled fictional data. No API keys are needed.

## Manual runs and the PDF

**Scheduled vs manual runs**

| Run | Limit | What it sends |
|---|---|---|
| Scheduled (08:00) | Once per day; re-runs never duplicate | The daily edition |
| Manual `live` (Run workflow button) | **Unlimited** | An **extra edition** ("🔄 অতিরিক্ত আপডেট") with any stories that are **new since the last edition** |

- A story is never sent twice. If nothing new has turned up, a manual run sends a short "সর্বশেষ ব্রিফিংয়ের পর নতুন কোনো প্রাসঙ্গিক খবর পাওয়া যায়নি" note, so you can see it worked.
- A manual run before 08:00 doesn't stop the scheduled edition; that one then carries only what's new.
- If an earlier edition today was interrupted, a manual run finishes delivering it first instead of starting another.
- **Cost:** a manual run only analyses *new* articles, so repeated runs are cheap. Each is still a few OpenAI calls (about $0.01–0.10, depending on how much is new).
- Locally: `python -m app run --manual`.

**The PDF**
- It's made with headless Chrome/Chromium from the same notes. GitHub's runners have Google Chrome, and the workflow installs the Noto fonts so Bengali renders correctly.
- It's sent **after** all text messages, per subscriber, and tracked like a message: a retry sends only what's missing, and if the file is gone on a later run it's rebuilt from the saved state.
- It's best effort. If it can't be made (no browser, font problem), the text edition is still sent and the run log/state records `pdf_error`.
- No PDF is sent for an empty "nothing new" edition or a failure notice.
- Turn it off with `python -m app run --no-pdf`; use a specific browser with the `CHROME_PATH` environment variable.
- The PDF HTML escapes all article and model text and has a Content-Security-Policy that blocks scripts and external loads.

---

## Architecture and data flow

```
        ┌──────────── GitHub Actions (cron 02:00 UTC = 08:00 Dhaka) ────────────┐
        │                                                                       │
 RSS ──▶│ A. collector.py   fetch feeds (robots.txt, timeouts, size caps)       │
 feeds  │        │          normalise URLs, parse dates safely                  │
        │ B. deduplicator.py  same URL / title / excerpt → one "event"          │
        │        │                                                              │
        │ C. screening.py   1 AI call: relevance + categories + grouping        │──▶ OpenAI API
        │        │          (model may only reference ids we gave it)           │    (prepaid)
        │ D. extractor.py   fetch article page, strip nav/ads/related blocks    │
        │        │          → full text, or labelled excerpt-only, or skip      │
        │ F. analysis.py    1 AI call per story → JSON note                     │──▶  (Gemini API if
        │      grounding.py drop facts whose quote isn't in the article and     │
        │        │          sentences with numbers not in the article           │
        │ E. scoring.py     weighted rubric → top 5–8 above threshold           │
        │ G. questions.py   1 AI call → MCQs, strictly validated                │──▶  OpenAI fails)
        │        │                                                              │
        │    render.py      Bengali, HTML-escaped, phone-friendly sections      │
        │    subscribers.py read /start & /stop (getUpdates), update the list    │◀─▶ Telegram
        │    telegram_delivery.py  chunk ≤3800 chars, send to every subscriber,  │──▶ Telegram
        │                   record message ids per subscriber                   │
        │        │                                                              │
        │    state.py       state/state.json (atomic, versioned), checkpointed  │
        │                   after every stage and every story                   │
        └────────┬──────────────────────────────────────────────────────────────┘
                 ▼
        git commit state/state.json → your repository (memory between runs)
```

Typical model usage per day: 1 screening + up to 9 analyses + 1 MCQ call ≈ **11 requests** (~70k input / ~25–45k output tokens), hard-capped at **16 per provider per run** including retries (`llm.max_calls_per_run`).

### Project layout

| Path | Purpose |
|---|---|
| `app/config.py` | Loads `config/settings.toml` + environment secrets |
| `app/sources.py` | Source registry (`config/sources.toml`) |
| `app/http.py` | Polite fetcher: robots.txt, timeouts, size caps, bounded retries |
| `app/collector.py` | Stage A — feeds → items |
| `app/deduplicator.py` | Stage B — duplicate detection, event grouping |
| `app/screening.py` | Stage C — relevance screening |
| `app/extractor.py` | Stage D — article text extraction, excerpt-only decision |
| `app/scoring.py` | Stage E — rubric scoring and ranking |
| `app/analysis.py`, `app/grounding.py` | Stage F — study notes and evidence checks |
| `app/questions.py` | Stage G — MCQ generation and validation |
| `app/render.py` | Telegram HTML rendering (Bengali) |
| `app/telegram_delivery.py` | Chunking, sending, retries |
| `app/subscribers.py` | `/start`, `/stop`, invite code, subscriber list |
| `app/state.py` | Versioned JSON state store |
| `app/pipeline.py` | Orchestration, checkpoints, idempotency |
| `app/llm/` | Provider interface + fallback chain: `openai.py`, `gemini.py`, `ollama.py` (local), `mock.py` (offline/tests) |
| `app/prompts.py` | Prompts with untrusted-content guards |
| `config/` | `settings.toml`, `sources.toml` |
| `state/state.json` | Created by the first live run, committed by the workflow |
| `sample_data/` | Fictional feeds/articles for offline runs and tests |
| `tests/` | Pytest suite (all external calls mocked) |

Dependencies: Python 3.12 (3.11+ works), `requests`, `feedparser`, `beautifulsoup4`; `pytest` for tests; Chrome/Chromium and a Bengali font (Noto) for the PDF, both present or installed on GitHub's runners. OpenAI and Gemini are called over plain REST (no SDKs), which keeps the provider layer swappable.

---

## Full setup guide

Allow about 20–30 minutes. Do the steps in order.

### Step 0: Make the repository private

The workflow commits `state/state.json`, which records what you read each day **and your subscribers' Telegram chat ids and first names**. It contains no credentials, but it must not be public.

GitHub → your repository → **Settings → General** → scroll to **Danger Zone** → **Change repository visibility** → **Make private**.

> Private repositories get 2,000 free Actions minutes per month on GitHub Free. This project uses about 100–200 (see [costs](#cost-and-free-tier-limits)).

### Step 1: Put the code on your default branch

GitHub only runs **scheduled** workflows from the repository's **default branch** (`main` here).

- If this code is on another branch (e.g. `claude/…`), open a Pull Request into `main` and merge it. If `main` doesn't exist yet, create it from that branch: **Code → branch dropdown → View all branches → New branch** (name `main`, source = the code branch). Then set it as default under **Settings → General → Default branch**.
- If you protect `main` with branch rules that block direct pushes, the workflow can't commit state. Either allow `github-actions[bot]` to bypass, or don't protect `main` (it's a personal repo).

### Step 2a: Create an OpenAI API key (primary model, prepaid)

1. Go to <https://platform.openai.com/>, sign in, open **Settings → Billing** and **add prepaid credit**. US$5–10 lasts about 1–4 months. A card that works for international online payments is needed.
2. **Turn auto-recharge OFF.** Then the most you can ever spend is the credit you added. When it runs out, OpenAI refuses requests and the app switches to Gemini automatically.
3. Optional extra guard: **Settings → Limits** (or the project's settings) → set a monthly budget/usage limit and an email alert.
4. **API keys → Create new secret key.** Copy it (starts with `sk-`); this is `OPENAI_API_KEY`.
5. **Check the model and price.** The default is `gpt-5.4-mini` (in `config/settings.toml`); a cheaper option is `gpt-5.4-nano`. I took these ids and prices from third-party trackers and couldn't confirm them on OpenAI's site from the build environment, so check <https://openai.com/api/pricing> and <https://platform.openai.com/docs/models>. To change the model without editing code, set the repository variable `OPENAI_MODEL` (Step 4). If the price differs, update `openai_usd_per_million_input/output` so the logged cost estimate stays right.
6. Data use: OpenAI states API inputs aren't used for training by default. Only public news text is sent.

### Step 2b: Create a Gemini API key (free fallback)

1. Go to **Google AI Studio → API keys**: <https://aistudio.google.com/apikey> and sign in.
2. Click **Create API key**. Prefer **a new Google Cloud project that has no billing account linked**.
   - Projects **without** billing stay on the **Free** tier. When the free quota runs out, requests fail with HTTP 429 and this app produces a partial edition. It never pays.
   - If you link billing to the project, it moves to a paid tier and usage could be charged. Don't, unless you intend to pay.
3. Copy the key; you'll paste it into GitHub in Step 4.
4. **Check the model.** The default is `gemini-3.8-flash` (set in `config/settings.toml`). Third-party pricing summaries from October 2026 list it as free-tier eligible, but **I could not confirm this against Google's own pages from the build environment**. Open <https://ai.google.dev/gemini-api/docs/pricing> and <https://ai.google.dev/gemini-api/docs/rate-limits> and confirm:
   - the model shows "Free of charge" in the Free Tier column, and
   - its free requests-per-day (RPD) limit is comfortably above ~16.

   If it isn't, choose a free Flash or Flash-Lite model and set it (Step 4, optional variable `GEMINI_MODEL`).
5. Privacy note: on the free tier, Google states prompts and responses may be used to improve its products. The app only sends public news text, never personal data.

### Step 3: Create the Telegram bot and choose an invite code

1. In Telegram, open **@BotFather** → send `/newbot` → choose a name and a username ending in `bot` (e.g. `bcs_daily_brief_bot`).
2. BotFather replies with a **token** like `123456789:AAH…`. That's `TELE_BOT_TOKEN`. Treat it like a password.
3. Optional but recommended: in BotFather, `/setcommands` → choose your bot → paste:
   ```
   start - ব্রিফিং চালু করুন
   stop - ব্রিফিং বন্ধ করুন
   help - সাহায্য
   ```
4. **Make up an invite code** (`BOT_INVITE_CODE`): 8–64 characters using only `A-Z a-z 0-9 _ -`, e.g. `bcs-friends-7Kq2xP`. Your **invite link** is then:
   ```
   https://t.me/<your_bot_username>?start=<your_invite_code>
   ```
   Only people who open the bot through this link can subscribe. Without a code, anyone who finds the bot could join (still capped at `max_subscribers` = 10).

There's no chat id to look up: the bot learns who to send to when people press **Start**. See [Subscribers](#subscribers-you-and-your-friends).

### Step 4: Add GitHub Secrets (and optional variable)

Repository → **Settings → Secrets and variables → Actions**.

**Secrets** tab → **New repository secret** for each:

| Name | Value |
|---|---|
| `OPENAI_API_KEY` | key from Step 2a (primary model) |
| `GEMINI_API_KEY` | key from Step 2b (free fallback) |
| `TELE_BOT_TOKEN` | token from Step 3 |
| `BOT_INVITE_CODE` | invite code from Step 3 (optional but recommended) |

Optional, on the **Variables** tab → **New repository variable**: `OPENAI_MODEL` and/or `GEMINI_MODEL` = a model id. These override the config file without a code change.

You can run with just one AI key: without `OPENAI_API_KEY` the app uses Gemini only (free); without `GEMINI_API_KEY` it uses OpenAI only (no fallback).

Secrets are masked in logs, and this app also redacts them from its own messages.

### Step 5: Enable GitHub Actions

1. **Settings → Actions → General → Actions permissions**: "Allow all actions and reusable workflows" (or at least allow GitHub-authored actions: `actions/checkout`, `actions/setup-python`, `actions/upload-artifact`).
2. Same page, **Workflow permissions**: the default ("Read repository contents") is fine. The workflow requests `contents: write` itself, and nothing else.
3. Open the **Actions** tab. If GitHub shows "Workflows aren't being run on this repository", click **I understand… enable them**.

### Step 6: Verify everything with manual runs (in this order)

**Actions → Daily BCS briefing → Run workflow** → pick a **mode** → **Run workflow**.

| # | Mode | What it proves | Uses quota / sends? |
|---|---|---|---|
| 1 | `offline` | Code, tests and rendering work | No API calls, nothing sent |
| 2 | `verify-sources` | Each feed is reachable, allowed by robots.txt and parses | Fetches feeds only |
| 3 | `check-model` | Both keys work and both model ids exist | 1 metadata call each (no generation, no cost) |
| 4 | `subscribers` | First, you (and your friends) open the **invite link** and press **Start**. Then this run registers you, sends each person a welcome message, and lists who's subscribed | Welcome replies only |
| 5 | `test-telegram` | Delivery works for every subscriber | 1 test message per subscriber |
| 6 | `dry-run` | Full real pipeline. Download the artifact to read the edition; the run summary shows tokens used and estimated cost | ~11 OpenAI calls (≈ $0.05–0.25), **nothing sent**, saved state untouched |
| 7 | `live` | The real thing | Sends an edition (text + PDF) to every subscriber, commits state. Run it **as often as you like**: each manual run sends whatever is new since the last edition |

After `verify-sources`, edit `config/sources.toml`: set `enabled = false` for any source that doesn't show `OK`, and update its `verification` note. See [News sources](#news-sources).

### Step 7: Let the schedule run

The cron line `0 2 * * *` runs daily at **02:00 UTC = 08:00 Asia/Dhaka** (Bangladesh is UTC+6 with no daylight saving). GitHub cron is always UTC.

GitHub doesn't guarantee the minute: scheduled runs often start several minutes late, sometimes much later at busy times, and occasionally get skipped. If one is skipped, run `live` manually. A re-run of the *scheduled* job never sends a duplicate; *manual* runs send extra editions with only new stories.

### Setup checklist

- [ ] Repository made **private**
- [ ] Code merged into the **default branch** (`main`)
- [ ] OpenAI credit added, **auto-recharge off**, API key created; model id and price checked
- [ ] Gemini key created in a project **without billing**; model's free-tier status confirmed on Google's pricing page
- [ ] Telegram bot created; invite code chosen
- [ ] Secrets `OPENAI_API_KEY`, `GEMINI_API_KEY`, `TELE_BOT_TOKEN`, `BOT_INVITE_CODE` added
- [ ] You and your friends opened the invite link and pressed **Start**
- [ ] Actions enabled
- [ ] Manual runs: `offline` → `verify-sources` (and adjust `sources.toml`) → `check-model` → `subscribers` → `test-telegram` → `dry-run` → `live`
- [ ] Next morning: briefing arrives and a `state: run for …` commit appears

---

## Subscribers: you and your friends

The bot sends the same edition to every subscriber. There's no fixed chat id and no always-on server.

**How someone joins**
1. You share the invite link `https://t.me/<your_bot>?start=<BOT_INVITE_CODE>` privately (e.g. in your friends' group chat).
2. They open it in Telegram and press **Start**.
3. The bot reads new messages at its **next run** (the 08:00 run, or right away if you run the `subscribers` mode), adds them, and replies "✅ … যুক্ত করা হয়েছে".
4. From the next edition on, they receive the daily briefing. Someone who joins after today's edition went out gets tomorrow's.

**Commands:** `/start` join · `/stop` leave · `/help`. A person who **blocks** the bot or deletes the chat is removed automatically, and that never stops delivery to the others.

**Timing caveat:** the bot only reads messages when the workflow runs, and Telegram keeps unread messages for **24 hours**. So a reply can take until the next run, and a Start pressed more than 24 hours before any run is lost. If a friend didn't get a welcome message, have them press Start again and run `subscribers` from the Actions tab.

**Groups:** you can also add the bot to a Telegram group and send `/start <code>` (or `/start@your_bot <code>`) there. The whole group then counts as one subscriber.

**Limits and privacy**
- At most `telegram.max_subscribers` (default **10**) active subscribers. Change it in `config/settings.toml`.
- Messages to each chat are spaced about 1 second apart (Telegram's per-chat limit). 5 friends × ~4 messages takes roughly 20 seconds.
- The state file stores each subscriber's chat id, first name (to show you who joined), join/leave time and per-edition delivery record. Nothing else. This is why the repository must be private.
- **To see who's subscribed:** run the `subscribers` or `status` mode. **To remove someone:** ask them to send `/stop`, or set their entry's `"status"` to `"inactive"` in `state/state.json` and commit. **To lock the list:** change `BOT_INVITE_CODE` (existing subscribers stay; old links stop working).

**Delivery tracking is per subscriber.** If sending fails for one person (e.g. a Telegram outage mid-run), the others are unaffected. The next run sends that person only the messages they're missing, and the edition is marked delivered once every active subscriber has it.

## Running locally

Only needed for development or testing; the cloud workflow doesn't need your computer.

```bash
git clone <your repo> && cd bcs-newspaper
python3.12 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt

python -m pytest                       # unit tests, all mocked
python -m app run --offline            # full sample edition, no keys needed → output/edition-*.txt

cp .env.example .env                   # then fill in real values
set -a; source .env; set +a            # load into the shell (bash/zsh)
python -m app verify-sources
python -m app check-model
python -m app subscribers              # after pressing Start via the invite link
python -m app test-telegram
python -m app run --dry-run            # real feeds + OpenAI (Gemini fallback), never sends; scratch state
python -m app status                   # summary of state/state.json
```

Avoid running `python -m app run`, `subscribers` or `test-telegram` locally *and* in Actions: each has its own copy of the state file, so they'd disagree about deliveries and subscribers (and compete for the same Telegram updates).

---

## Manual runs and modes

| Command | Effect |
|---|---|
| `python -m app run` | Register new subscribers, build today's edition and deliver it (text + PDF) to all of them. No-op if today's edition was already delivered. Skips building (no model calls, no cost) when nobody has subscribed. |
| `python -m app run --manual` | Extra edition with whatever is new since the last one; can run any number of times (the workflow's manual `live` uses this). |
| `python -m app run --no-pdf` | Don't create or send the PDF. |
| `python -m app run --dry-run` | Real feeds + real model, **never sends**. Uses `output/dry-run-state.json` (fresh each time), so production state is untouched. |
| `python -m app run --offline` | Bundled fictional sample data + mock model. No network, no keys. Implies `--dry-run`. |
| `python -m app run --date 2026-10-24` | Override the edition date (testing). |
| `python -m app run --provider openai\|gemini\|ollama\|mock` | Use only that provider (no fallback). |
| `python -m app subscribers` | Read new `/start` / `/stop` messages now, reply, and list active subscribers. |
| `python -m app test-telegram` | Same registration step, then one test message to each subscriber. |
| `python -m app verify-sources` | Live check of every configured feed. Exit code 1 if an enabled feed fails. |
| `python -m app check-model` | Confirms each configured model id works with its key (not proof of credit/quota). |
| `python -m app status` | Last edition, events by status, source health, last run. |

Exit codes: `0` success / dry run / already delivered; `1` failed; `2` failed but a failure notice was already sent to Telegram; `3` configuration error.

---

## News sources

Sources live in `config/sources.toml`. Each entry has a name, homepage, `feed_url` (only if verified), type, subject relevance, `enabled`, `fetch_articles`, timeout, retry limit and a `verification` note.

**Feed verification status at build time.** The build environment couldn't reach news sites directly, so none of these has been fetched live yet. Status is based on public evidence:

| Source | Feed | Evidence | Shipped as |
|---|---|---|---|
| Prothom Alo | `https://www.prothomalo.com/feed` | Used by your working prototype | enabled |
| Dhaka Tribune | `https://www.dhakatribune.com/feed/` | Self-link inside search-indexed Dhaka Tribune RSS documents | enabled |
| Dhaka Tribune Business | `https://www.dhakatribune.com/feed/business` | Search-indexed RSS 2.0 document observed | enabled |
| The Daily Star | `https://www.thedailystar.net/frontpage/rss.xml` | Listed in public OPML feed directories | enabled |
| UN News Asia-Pacific | regional feed URL pattern | Pattern only; exact feed unconfirmed | **disabled** |
| BSS, Bangladesh Bank, BBS, MoFA | none | No official feed found | reference only (never fetched) |

**Run `verify-sources` (Step 6) and disable anything that fails.** A failing source never stops the others. Its health is tracked in state, and after 3 consecutive failures the edition footer shows a maintenance notice.

**Official sources.** Government sites without feeds are kept as reference entries with `aliases`. When an article names an institution (e.g. "Bangladesh Bank" / "বাংলাদেশ ব্যাংক"), the story links that institution's homepage under "দাপ্তরিক সূত্রে যাচাই" so you can check the primary source. The app doesn't scrape government sites.

**Adding a source.** Find the feed link on the publisher's site (often in the page footer or a `<link rel="alternate" type="application/rss+xml">` tag), add a `[[source]]` block with `enabled = false`, run `verify-sources`, then enable it if it reports `OK`. Never guess feed URLs.

**Access rules.** The fetcher uses an honest user agent, obeys `robots.txt`, never retries 401/403/451, and never bypasses paywalls or anti-bot protection. If a page is blocked, the story uses a clearly labelled **excerpt-only** analysis (when the feed excerpt is substantial) or is skipped.

---

## Factual grounding: how hallucinations are kept out

The model writes the notes, but deterministic code decides what you see:

- **Every key fact needs a verbatim quote** from the retrieved text. Facts whose quote isn't found in the article are dropped.
- **Numbers are checked.** Every number in a fact must appear in its quote. Any sentence in the summary, analysis or background that contains a number not present in the article is removed. (Bengali and English digits are normalised before comparison.)
- **Claims are separated from facts.** "X said …" goes under *উদ্ধৃত দাবি (যাচাইকৃত তথ্য নয়)*.
- **Analysis is labelled** *(AI বিশ্লেষণ)*; background is labelled *(প্রতিবেদনের বাইরের সাধারণ প্রেক্ষাপট)* and may not contain figures.
- **Excerpt-only stories are marked** ⚠️ and analysed only from the excerpt.
- A story with **no verifiable fact is excluded**, not padded.
- **MCQs** must cite a verified fact; the correct option's numbers must appear in that fact. Each needs exactly 4 distinct options, a valid answer index, no duplicate question and no reused fact. Fewer than 3 valid MCQs → fewer are sent, with a note.
- **Screening ids are validated.** The model can only reference candidate ids we supplied; unknown ids are ignored.
- **Neutrality.** Prompts require political neutrality and forbid inferring motives.

The scoring weights (syllabus 30%, significance 25%, facts 20%, written potential 15%, novelty 10%) are **starting heuristics, not validated predictions of exam questions**. Change them in `config/settings.toml`; they must sum to 1.0.

---

## Partial editions and quota exhaustion

The model can become unavailable mid-run (daily quota used up, per-minute limit, outage, retired model). The app treats each error type differently:

| Situation | Behaviour |
|---|---|
| HTTP 429, per-minute limit | Wait the server's suggested delay (max 90 s), retry up to 3 times |
| HTTP 429, daily quota or `limit: 0` | Stop calling the model for this run, immediately |
| 5xx / network error | Exponential backoff with jitter, max 3 retries |
| Malformed JSON / blocked / truncated output | Retry once, then count a failed attempt for that story |
| 401/403/invalid key, 404 unknown model | Permanent: no retries, run stops using the model |
| Own budget `max_calls_per_run` reached | Treated like quota exhaustion |

When this happens mid-run:

1. Every completed analysis is **already saved** (state is checkpointed after each story).
2. Completed, useful stories are sent as an edition labelled **⚠️ আংশিক সংস্করণ**, with the reason and the number of candidate stories still waiting.
3. Missing stories are never invented.
4. If nothing usable was produced, every subscriber gets one short **failure notice** instead (only once per day).
5. Unfinished stories stay queued. The next run (scheduled tomorrow, or a manual `live` run after a failure) continues where it stopped: no re-screening, no re-analysis of finished work. Quota failures don't count against a story; other failures are retried at most `max_attempts_per_story` (3) times.
6. Stories older than `freshness_hours` (36 h) expire rather than piling up.

---

## State persistence and recovery

`state/state.json` (schema version 2) records: processed articles (canonical URL, content hash, publication and retrieval times), event groups, per-stage status (`new → selected/maybe/rejected → analysed/analysis_rejected/failed → included`), stored analyses, attempt counts and last error, editions (status, partial flag, MCQs, every Telegram chunk with message id and sent time), last successful edition, source health, and the last 30 runs.

- **Atomic writes:** temp file → fsync → rename. The previous version is kept as `state.json.bak` during a run.
- **Validation on startup:** malformed records are dropped individually; a corrupt file is preserved as `state.json.corrupt-<time>` and recovered from the backup, or (last resort) a fresh state is started and the run log says so.
- **Migration:** older schemas are upgraded in steps (v1 → v2 added subscribers). A file from a **newer** schema is never overwritten; the run stops instead.
- **Subscribers:** chat id, first name, type (private/group), active/inactive, join/leave time, plus the last processed Telegram `update_id` so no `/start` is handled twice.
- **Idempotent delivery:** an edition is rendered once and its chunks saved *before* sending. For each subscriber, each chunk's Telegram message id is saved as soon as Telegram confirms it. An edition is marked `delivered` only when **every active subscriber** has **all** chunks. The PDF counts as the last item. A retry sends each person only what they're missing. The scheduled edition for a date is never sent twice; manual extra editions are stored as `<date>-m1`, `-m2`, … and never repeat a story.
  - Residual risk: if the runner dies in the instant between Telegram accepting a message and the state write, that one chunk could be sent twice on retry. Telegram has no idempotency key to prevent this.
- **Git:** the workflow commits `state/state.json` with the workflow token (`contents: write` only). On a rejected push it rebases and retries; if the state file itself conflicts it **stops without overwriting** newer state and keeps this run's state in the workflow artifact. It never force-pushes. A `concurrency` group ensures only one run touches state at a time.
- **If a state commit fails,** the edition was still delivered (or saved in the artifact), the job turns red, and you get a Telegram notice. Download the artifact to recover the state if needed.
- Old records are pruned after `retention_days` (14), editions after 30 days.

To **reset** everything: delete `state/state.json` and commit. The next run starts fresh, so it may re-send stories it had already shown you.

---

## Logs and troubleshooting

- **Actions tab → Daily BCS briefing → a run → `briefing` job:** step-by-step logs. The "Run" step prints a JSON summary (`outcome`, `stories`, `mcqs`, `partial`, `llm_calls`, `errors`).
- **Artifacts** (bottom of the run page, kept 7 days): `output/edition-<date>.txt` (exactly what was sent), `edition-<date>.json`, `run-result.json`, and the state file.
- **`status` mode** (or `python -m app status`): last edition, events by status, per-source health, last run errors.
- **Email:** GitHub emails you (the repository owner) with details when a scheduled run fails. Subscribers get a short, non-technical Telegram notice.

| Symptom | Likely cause / fix |
|---|---|
| `Telegram rejected request (401)` | Wrong `TELE_BOT_TOKEN` |
| Briefing arrives but no PDF | Check the run log for `PDF could not be created` / `pdf_error`. Usually no Chrome or the font step failed; the text edition is unaffected |
| Run says `no_subscribers` / "No subscribers yet" | Nobody has pressed Start yet, or their Start is older than 24 h → open the invite link, press Start, run `subscribers` |
| Friend pressed Start but got no welcome | Replies only happen at the next run → run `subscribers`. Got "❌ আমন্ত্রণ লিংক… সঠিক নয়" → they used the plain bot link, not the invite link |
| "সদস্যসংখ্যা পূর্ণ" reply | `max_subscribers` reached → raise it in `config/settings.toml` |
| `A webhook is set for this bot` | Another tool set a webhook → open `https://api.telegram.org/bot<TOKEN>/deleteWebhook` once |
| A subscriber marked inactive unexpectedly | They blocked the bot or left the group → they can open the invite link and press Start again |
| `Provider openai disabled for this run: … credit exhausted` | Prepaid OpenAI credit used up → today ran on Gemini; add credit when you want OpenAI back |
| `OpenAI rejected credentials` / `OpenAI model unavailable` | Bad `OPENAI_API_KEY`, or model id wrong/retired → fix the secret or set `OPENAI_MODEL`. Gemini covered the run meanwhile |
| `OpenAI request failed (400): … reasoning_effort` | Your model doesn't accept that parameter → set `openai_reasoning_effort = ""` |
| Edition footer says "বিকল্প মডেল" was used | OpenAI was unavailable for part of the run; check the log line `Provider openai disabled …` |
| `Gemini model unavailable` | Model id retired/renamed → set `GEMINI_MODEL` to a current free model |
| `Gemini quota exhausted … limit: 0` | Model has no free quota for your project → pick another model |
| `Gemini rejected credentials` | Bad or revoked `GEMINI_API_KEY` |
| `Source X failed: HTTP 403` / `robots.txt disallows` | Publisher blocks automated access → disable that source |
| `Source X failed: Malformed feed` | Feed URL wrong/moved → re-verify |
| Many stories ⚠️ excerpt-only | Site blocks article pages or renders with JavaScript; acceptable, or set `fetch_articles = false` |
| No briefing at 08:00 | Scheduled run delayed/skipped by GitHub, workflow not on default branch, or Actions disabled → check the Actions tab, run `live` manually |
| `state/state.json changed on the remote too` | Someone edited state manually while a run was going → resolve by hand, never force-push |
| Exit code 3 / "Configuration error" | Typo in `config/*.toml`; the message names the key |

---

## Cost and free-tier limits

Daily usage: about 11 model requests ≈ **70k input + 25–45k output tokens** (Bengali output, plus hidden "reasoning" tokens on newer models, which are billed as output). The edition is built once, so the number of subscribers doesn't change the AI cost.

| Model (primary) | Price per 1M tokens (in / out)* | Per day | Per month |
|---|---|---|---|
| `gpt-5.4-nano` | $0.20 / $1.25 | ~$0.05–0.07 | **~$1.5–2 (≈ ৳180–260)** |
| `gpt-5.4-mini` (default) | $0.75 / $4.50 | ~$0.17–0.26 | **~$5–8 (≈ ৳600–1,000)** |
| Gemini fallback (free tier) | free | 0 | 0 |

\*Third-party snapshots from mid-2026, at ≈ ৳120/US$. Verify on <https://openai.com/api/pricing>. Dry runs cost about one day's usage each.

**See the real numbers:** every run logs `llm_usage` (requests and tokens per provider) and `estimated_openai_cost_usd` in its JSON summary, and saves them in `state/state.json` under `runs`. Compare with OpenAI's **Usage** page.

**How spending is bounded:**
- OpenAI is prepaid. With auto-recharge off, spend can never exceed the credit you added.
- Each provider is capped at `llm.max_calls_per_run` (16) requests per run, retries included.
- When OpenAI credit runs out, the run continues on Gemini's free tier. Gemini free-tier projects without billing just refuse requests over quota (HTTP 429), and the app makes a partial edition. Nothing upgrades to a paid plan automatically.
- To go back to **৳0**, remove the `OPENAI_API_KEY` secret (or set `provider = "gemini"`); the app then runs on Gemini only.

| Service | What's used | Free allowance (verify yourself) |
|---|---|---|
| GitHub Actions (private repo, GitHub Free) | ~3–6 min/day → 100–200 min/month | 2,000 min/month; 500 MB artifact storage |
| Gemini Developer API (Free tier) | only on fallback days | Per-model RPM/RPD limits per project; resets at midnight Pacific time |
| Telegram Bot API | ~4 messages × subscribers per day, plus replies | Free (rate limits: ~1 msg/s per chat, handled automatically) |
| News feeds | a few HTTP requests/day | Free; subject to each publisher's terms |

## AI providers: OpenAI with Gemini fallback

`config/settings.toml`:
```toml
[llm]
provider = "openai"              # primary
fallback_providers = ["gemini"]  # tried in order; [] to disable
openai_model = "gpt-5.4-mini"
gemini_model = "gemini-3.8-flash"
```

**When the run switches to Gemini.** Each provider has its own call budget, spacing and retries. OpenAI is switched off **for the rest of that run** (and Gemini takes over) when:
- the prepaid credit is exhausted (`429 insufficient_quota`),
- the key is rejected or the model id is unknown (401/403/404),
- it keeps failing after its retries (outage, 5xx, timeouts),
- it hits the per-run call budget.

Short per-minute rate limits are waited out and retried on OpenAI first. If OpenAI returns unusable output for one request, that single request is retried on Gemini, and OpenAI stays in use. The next day's run tries OpenAI again.

**What readers see.** If a fallback was used, the edition footer says so ("…বিকল্প মডেল দিয়ে করা হয়েছে"). The fact checks (verbatim quotes, numbers present in the article, MCQ validation) apply identically to every model's output. If both providers are unavailable, you get the usual partial edition or failure notice; nothing is invented.

**Other combinations:** Gemini only (`provider = "gemini"`, `fallback_providers = []`, or just don't set `OPENAI_API_KEY`). OpenAI only (`LLM_FALLBACK=none`). A local model (see below) can also be primary or fallback.

## Disabling the schedule

- **Pause:** Actions → *Daily BCS briefing* → **⋯** (top right) → **Disable workflow**. Re-enable the same way.
- **Remove the schedule but keep manual runs:** delete the two `schedule:` / `- cron:` lines in `.github/workflows/daily-briefing.yml`.
- **Stop completely:** delete the workflow file, and optionally revoke the OpenAI and Gemini keys and the bot (BotFather → `/revoke`).

Note: GitHub automatically disables schedules in **public** repositories after 60 days without activity. The daily state commit counts as activity.

---

## Switching to a local model

All AI calls go through `app/llm/base.py`:

- `LLMProvider.generate_raw(request) -> str` is the only thing a provider implements. It maps its failures to `QuotaExhaustedError`, `RateLimitError`, `TransientLLMError`, `PermanentLLMError`, `ModelUnavailableError` or `InvalidResponseError`.
- `LLMClient` adds the call budget, spacing, retries, backoff and JSON parsing for every provider.
- Prompts, validation, grounding and rendering don't care which provider produced the JSON.

**Ollama (included, experimental, not tested against a real Ollama server):**

```bash
ollama pull qwen2.5:7b-instruct          # any instruction model that follows JSON well
export LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:7b-instruct
python -m app run --dry-run              # check quality before going live
```

GitHub-hosted runners **can't reach a model on your computer**. Options:
1. Run the pipeline on your own machine on a schedule (cron / Task Scheduler) with `python -m app run`. The machine must be on at 08:00, and you should stop the GitHub workflow so the two don't keep separate state.
2. Register your machine as a **self-hosted runner** (free) and change `runs-on` to `self-hosted`. Same requirement: the machine must be on.

Small local models are weaker at following the strict JSON and Bengali instructions. The grounding checks will discard more of their output, so expect more partial editions. Compare with a `--dry-run` first.

**Another provider:** add `app/llm/<name>.py` with a `generate_raw` implementation and one branch in `build_provider()` in `app/llm/__init__.py`. Nothing else changes.

---

## Testing

```bash
python -m pytest                 # ~100 unit tests, fully mocked, < 2 s
```

A test fixture blocks real HTTP, so unit tests can never spend OpenAI credit or Gemini quota, or message Telegram. Coverage includes feed parsing with malformed items, missing/invalid/future dates, invalid URLs, duplicate detection, extraction failure and excerpt-only rules, the scoring rubric, malformed model output, unsupported facts and numbers, MCQ schema/answer-index validation, HTML escaping and chunking, Telegram retries and plain-text fallback, Gemini and OpenAI 429/404/5xx classification (including OpenAI `insufficient_quota`), the OpenAI → Gemini fallback chain (credit exhausted, bad key/model, outage, invalid output, both exhausted, mid-run switch completing a full edition, cost estimate), retry limits and call budget, state corruption/backup recovery/migration/newer-schema refusal, quota exhaustion mid-run with resumption, interrupted delivery resuming without duplicates, failure-notice de-duplication, low-relevance days without padding, dry-run behaviour, secret redaction in logs, and subscribers: invite-code checks, `/stop`, the cap, blocked users, group commands, malformed updates, per-subscriber delivery and resumption, and "no subscribers spends no quota".

**Live integration tests** (opt-in, real network; 1 model lookup and 1 Telegram `getMe`; no message is sent):

```bash
RUN_INTEGRATION=1 GEMINI_API_KEY=… TELE_BOT_TOKEN=… python -m pytest -m integration
```

The scheduled workflow runs the unit tests before each live run; if they fail, no briefing is sent and you're notified.

---

## Security and privacy

- **No secrets in the repository.** Credentials come only from environment variables / GitHub Secrets. `.env` is git-ignored; `.env.example` holds placeholders.
- **Subscriber data** (chat ids, first names) lives only in the private repository's state file. The invite code is compared in constant time and redacted from logs like the other secrets.
- **No secrets in logs.** The OpenAI and Gemini keys travel in headers, never URLs. A logging filter redacts the credentials from every log line, and urllib3 debug logging (which would print Telegram's token-bearing URL) is suppressed.
- **Web content is untrusted.** Article text is wrapped in delimiters the article can't forge, the model is told to treat it as data, and the model has no tools. Its output is parsed as JSON and validated. URLs are stripped from model text, all text is HTML-escaped, and links only ever come from our own feed records. A hostile article can at worst get itself rejected.
- **Network hygiene.** Timeouts on every request, size caps (3 MB) on feeds and pages, robots.txt respected, honest user agent, no paywall or anti-bot bypassing.
- **Least privilege.** The workflow token has `contents: write` only, used solely to commit `state/state.json`. No force-push.
- **Data use.** OpenAI states API data isn't used for training by default. On Gemini's free tier (fallback), Google may use prompts/responses to improve its products. Only public news text is sent to either.

---

## Configuration reference

`config/settings.toml` (non-secret; edit and commit):

| Key | Default | Meaning |
|---|---|---|
| `edition.max_stories` / `min_target_stories` | 8 / 5 | Target range. A ceiling, not a quota |
| `edition.min_story_score` | 55 | Stories below this (0–100) are never included |
| `edition.freshness_hours` | 36 | Items older than this are ignored or expire |
| `edition.mcq_count` | 3 | MCQs per complete edition |
| `extraction.min_article_chars` | 600 | Below this, the body counts as unavailable → excerpt-only |
| `extraction.max_article_chars` | 12000 | Text sent to the model is truncated here |
| `extraction.min_excerpt_chars_for_analysis` | 220 | Below this, an excerpt-only story is skipped |
| `analysis.max_analyses_per_run` | 9 | Model calls for analysis per run |
| `analysis.max_attempts_per_story` | 3 | Then the story is abandoned |
| `scoring.weights.*` | 0.30/0.25/0.20/0.15/0.10 | Rubric weights, must sum to 1 |
| `llm.provider` | `openai` | Primary: `openai`, `gemini`, `ollama` or `mock` |
| `llm.fallback_providers` | `["gemini"]` | Tried in order; override with `LLM_FALLBACK` (`none` to disable) |
| `llm.openai_model` | `gpt-5.4-mini` | Override with `OPENAI_MODEL` |
| `llm.openai_reasoning_effort` | `""` | Optional `low`/`minimal` to cut reasoning tokens (cost) |
| `llm.openai_usd_per_million_input/output` | 0.75 / 4.50 | Only for the logged cost estimate |
| `llm.gemini_model` | `gemini-3.8-flash` | Override with `GEMINI_MODEL` |
| `llm.max_calls_per_run` | 16 | Hard cap per provider, including retries |
| `llm.min_seconds_between_calls` | 6 | Request spacing |
| `telegram.max_message_chars` | 3800 | Chunk size (Telegram's limit is 4096) |
| `telegram.max_subscribers` | 10 | Cap on active subscribers |
| `telegram.min_seconds_between_messages` | 1.1 | Pause between messages to the same chat |
| `state.retention_days` | 14 | Pruning window |

Environment variables: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `TELE_BOT_TOKEN`, `BOT_INVITE_CODE` (secrets); `OPENAI_MODEL`, `GEMINI_MODEL`, `LLM_PROVIDER`, `LLM_FALLBACK`, `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `BCS_STATE_PATH`, `BCS_OUTPUT_DIR`, `BCS_CONFIG_DIR` (optional overrides).

Syllabus reference: subject mapping follows the BPSC BCS syllabus (preliminary subjects: Bangladesh affairs, international affairs, geography/environment/disaster management, general science, computer & ICT, ethics/values/good governance; written papers 005 Bangladesh Affairs, 007 International Affairs, 010 General Science & Technology). Always confirm the latest syllabus at <https://bpsc.gov.bd>.
