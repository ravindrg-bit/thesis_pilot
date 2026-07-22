# Methodology notes — running log

Working record of design decisions and empirical findings surfaced while building the
data-sourcing pipeline. Feeds the methods chapter, the version-drift discussion, and the
limitations section. Complements `thesis_config.py` (the *what*) and `data_provenance.md`
(dataset lineage). This is a **living document** — entries are updated as the build proceeds.

**Status legend:** _confirmed_ (observed live) · _provisional_ (to confirm) · _deferred_
(handled later, at silver or scale-up).

_Last updated: 31 May 2026._

---

## 1. Determinism and version control

**Determinism comes from repetition, not sampling controls.** `temperature = 0` and fixed
seeds are not available on this model generation: GPT-5.5 is a reasoning model (effort, not
temperature) and Gemini 3.x recommends keeping temperature at its default and removing
`temperature`/`top_p`/`top_k`. The determinism control is therefore the repeated-measures
design (k = 8) and the per-query coefficient of variation — consistent with V3 §7.2 and the
stochasticity literature (Atil et al. 2025; Yuan et al. 2025; Schulte et al. 2026, whose
7–8-run minimum sets k). Where a reasoning-effort knob exists we fix and record it; otherwise
defaults are used and the variance is reported. _Confirmed._

**Pin and log the *served* model id.** Every call records the model id the API actually
returned, not only the requested one, because the two can differ and drift over the
collection window — three of the six engines are on preview/recent releases. Dated snapshots
are pinned where offered (OpenAI `gpt-5.5-2026-04-23`); where not (Perplexity, Gemini, Kimi,
Mistral), the served-id log plus a tight collection window carry the version-drift control
(V3 §7.4). _Confirmed._

**Tool/mode version can silently change the citation channel.** Concrete case (Claude): the
dynamic-filtering web-search tool `web_search_20260209`, used standalone, produced *no*
structured inline citations — the model wrote sources as prose instead — whereas the previous
version `web_search_20250305` emitted them reliably. So the exact *tool* version, not just the
model version, must be pinned and logged; "latest/preview" tool versions are a reproducibility
hazard. _Confirmed._

## 2. What counts as a citation

**Grounding is a precondition.** A plain model call returns no sources, so PAWC and the AIS
verifiability rate have nothing to measure. Every engine is queried with its grounding
mechanism enabled (mechanism differs per engine — see §3). _Confirmed._

**Consulted ≠ cited.** Engines distinguish sources they *read* from sources they *credit*. The
visibility metrics use the *credited* (inline-cited) set; the *consulted* set (e.g. OpenAI's
`sources` list, Claude's `web_search_tool_result` blocks) is closer to "what was in context."
Both counts are recorded at silver — the gap between them, and the fact that engines differ in
citation propensity, is itself a finding (RQ3). _Confirmed._

**Selective grounding is a measured behaviour, not a defect.** Perplexity grounds every query;
OpenAI, Gemini, and Claude decide per query whether to search, and may search yet decline to
inline-cite (observed repeatedly with Claude on fact-seeking queries). "Searched but did not
cite" is a distinct, recordable state and a cross-engine signal for RQ3. We never force search
or force citations — that would measure forced behaviour, not native behaviour. _Confirmed._

**AIS verifiability is defined at cited-source-SET granularity, uniformly across all engines.**
For each verifiable claim, AIS asks whether it is supported by at least one source the engine
cited — judged by a rater fetching and reading the cited source, not by trusting any provider's
claim-citation pairing (the pairing's validity is the very thing under test). This common
granularity is required regardless of Kimi: the engines already cite at different native
granularities (per-span for ChatGPT/Claude/Gemini; whole-answer source list for Perplexity), so a
comparable protocol must operate at the level the coarsest engine supports. Defining AIS this way
(cf. citation-recall measures, Gao et al. 2023) lets Kimi's self-reported source set be judged on
identical terms. Per-span pairings are retained at silver for ChatGPT/Claude/Gemini as an optional
supplementary analysis. The construct AIS does NOT measure — citation *faithfulness* (whether the
answer causally derived from the listed sources) — is out of scope for the four metrics; for Kimi
it is additionally unverifiable (encrypted retrieval) and noted as such. _Methods (measurement) /
limitations._

## 3. Per-engine citation channels (verified 30 May 2026)

| Engine | Grounding mechanism | Citations live in | Decides to search? | Status |
|---|---|---|---|---|
| ChatGPT (gpt-5.5) | Responses API + `web_search` tool | inline `url_citation` annotations (+ separate `sources` list) | yes | confirmed |
| Claude (sonnet-4-6) | Messages API + `web_search_20250305` | `web_search_result_location` on text blocks (`cited_text`/title/url) | yes | confirmed (see §1) |
| Gemini (3.5-flash) | `generate_content` + `google_search` | `groundingMetadata.groundingChunks` | yes | confirmed (see §4) |
| Perplexity (sonar) | inherent (always grounds) | top-level `search_results` (title/url/date) | always | confirmed |
| Kimi (k2.6) | `$web_search` loop; **no structured channel** -> citations prompt-elicited from answer text | self-reported URLs (parsed from prose; flagged `self_reported_prompt_elicited`) | yes | confirmed |
| ~~DeepSeek (v4)~~ | none via API (search is app-only) | n/a | n/a | excluded (see §4) |

The citation shape is never assumed: the per-engine adapter is the only engine-specific code
(V3 §7.3), and each adapter was written only after inspecting a real verbatim response
(bronze-first). The canonical record is the uniform output every metric reads — the design
answer to cross-engine inconsistency (Puerto et al. 2025).

## 4. Engine-specific data caveats

**DeepSeek excluded.** DeepSeek's API is a stateless chat endpoint with no native web search
(search exists only in its consumer app), so it cannot produce native citations; including it
would leave PAWC/AIS structurally empty. Dropped 30 May 2026, taking the set from seven to six;
China remains represented via Kimi. _Decision; limitations / engine-selection._

**Kimi: no provider-certified citation channel; sources recovered by prompt elicitation.** All
of Moonshot's API surfaces were tested live (31 May 2026): the built-in `$web_search` returns an
opaque `search_id`; the Formula API (`moonshot/web-search:latest`) returns `encrypted_output`
(server-side, opaque) plus a `references` tool-call echo. None exposes a source URL. The model
holds the sources server-side, so citations are instead elicited into the answer text via a
citation-instructing system prompt (the consumer app's approach, via the official API) and parsed
out. These URLs are **self-reported (prompt-elicited), not provider-certified**, and every Kimi
`CitedSource` is flagged `self_reported_prompt_elicited`; the system prompt is logged in bronze
`request_params` for auditability. Validated on two query types (current-events and medical), Kimi
yields source counts comparable to the structured engines (e.g. 8 vs Claude's 7 on the same
medical query). This is a transparency finding — a frontier engine that grounds opaquely — and the
extraction asymmetry is disclosed, not hidden. _Methods / limitations._

**Gemini redirect URLs.** Gemini's grounding chunks give a `vertexaisearch.cloud.google.com`
redirect as the URL and the publisher *domain* in the `title` field (not a page title). The
domain is taken from `title`; resolving the redirect to the true publisher URL is deferred.
_Deferred (silver / scale-up)._

**URL canonicalisation.** Citation URLs carry provider tracking parameters (e.g.
`?utm_source=openai`), which would fragment domain/URL counts. Bronze keeps URLs verbatim;
silver strips tracking parameters before counting. _Deferred (silver)._

**Embedded-year drift in queries.** Some GEO-Bench queries contain a year (e.g. "art
collaborations 2023"); asked in 2026 they may surface different sources than when the benchmark
was built. A real-world-drift point alongside source-link attrition (V3 Stage 1). _Limitations._

## 5. Cross-region design and the language confound

Six engines span three regions — US (ChatGPT, Claude, Gemini, Perplexity), China (Kimi),
Europe (Mistral) — strengthening the generalisability angle of RQ3. Kimi reads a partly
Chinese-language web, a potential source-web/language confound; queries are issued in a single
language (English), engine region is logged per call, and any systematic regional effect is
reported rather than controlled away. _Methods (design) / limitations._

Why we chose to run each query 7 times

Every time you ask an AI engine the same question, you can get a slightly different answer. That's true even when you set every option to make the answer as consistent as possible — the engines themselves are not perfectly deterministic. So if you only ask a question once and measure that single answer, you don't actually know what the engine "typically" does. You just know what it did that one time. To recover the engine's typical behaviour, you have to ask the same question several times and average the results. The question is: how many times is enough?

The published literature in this field — specifically Schulte and colleagues (2026) — found that seven to eight repeats per question is the minimum needed to produce reliable measurements. Below that, the measurements are too noisy to trust. Above that, you're paying for extra runs without gaining much extra reliability. We chose to use k=7, which sits at the lower end of the recommended range.
Why we checked this with our own data

Rather than just trusting the literature figure, we ran an additional analysis on our pilot data to see whether seven repeats was actually necessary for our specific five engines. The technique is called bootstrap convergence analysis. In plain terms, it asks: how much does the measurement still wobble at one repeat, at two, at three, and so on? If the wobble drops sharply with each extra repeat and then flattens out, you've found the point where more repeats stop helping. That's the convergence point.
What the analysis showed

The wobble was still meaningfully decreasing across the full range we could test. This means each additional repeat continues to deliver real precision — there's no point at which adding another run would have been wasteful. We also found that one engine, ChatGPT, was substantially more variable than the other four. This was partly because ChatGPT often decides not to search the web for evergreen factual questions, producing zero-valued measurements on those queries, but also because when it does answer, its answers vary more from run to run than the other engines.
The conclusion we can defend

Our own analysis supports the literature recommendation. Seven repeats is the right number for our study: it sits within the range where each repeat is still buying us measurement precision, and it matches the published minimum from Schulte's work. Running fewer repeats would mean reporting measurements we couldn't trust; running many more would cost considerably more without meaningfully improving the result. For the most variable engine in our set, ChatGPT, the analysis shows that even seven repeats will not fully eliminate variability — and we will report that residual instability as a genuine finding about how that engine behaves, rather than as a flaw in our measurement.
