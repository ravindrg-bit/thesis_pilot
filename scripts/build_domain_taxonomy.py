"""
build_domain_taxonomy.py — classify every cited domain into an institutional
"domain-type" taxonomy and emit joinable enrichment artefacts.

Supplementary analysis for the Discussion chapter: answers the practitioner
question "what kinds of sources am I competing against on each engine?".

TWO-PHASE classification:

  PHASE 1 (rule-based) labels each cited domain as one of six high-precision,
  mutually-exclusive institutional types via TLD/suffix patterns + curated
  pay-level-domain lists (no API, no invented numbers):

      wikipedia | academic | government | news_media | reference | ugc_forum

  Everything else falls into a residual `commercial` bucket (~65% of citations
  — too coarse to be analytically useful).

  PHASE 2 (LLM) assigns the commercial residual a single-word granular industry
  label (cybersecurity | cloud | saas | ai | fintech | ecommerce | healthcare |
  edtech | ... | other) using Claude Haiku 4.5 — a model NOT in the five-engine
  study set (ChatGPT, Claude, Gemini, Kimi, Perplexity), to avoid circularity.

  PHASE 3 adds granularity to three coarse buckets:
    3a  academic   -> journal | preprint | database | university | research   (rules)
    3b  ugc_forum  -> forum | code | video | blogging | social | community     (rules)
    3c  other      -> blog | directory | marketplace | utility | retail |
                      agency | government_services | other                      (LLM)
  Phase 3c offers only the new long-tail labels (never the industries), so every
  Phase-2 industry category keeps an identical citation count.

  Deterministic (temperature 0); a single PLD->label cache holds the final LLM
  label for each commercial domain so the whole taxonomy reproduces with no API:

      python scripts/build_domain_taxonomy.py --skip-llm --write

Inputs  (silver):
    data/scaleup/silver/citations_scaleup.parquet   (58,851 rows; has own `domain`)
    data/scaleup/silver/sources_scaleup.parquet     (column-selective read ONLY;
                                                      file is ~3.5 GB)

We classify on citations.`domain` (the authoritative per-citation domain that is
already in the citations table). citations.`domain` and sources.`domain` differ
for ~26% of rows, but the difference is purely a www-prefix / extraction quirk
(e.g. `forbes.com` vs `www.forbes.com`) — both collapse to the same pay-level
domain and therefore the same domain_type. Classifying citations.`domain`
directly sidesteps that quirk and avoids a heavy join against the 3.5 GB sources
file.

Outputs (data/scaleup/enrichment/):
    domain_taxonomy.parquet            one row per unique domain (domain, domain_type, pay_level_domain)
    citations_scaleup_enriched.parquet citations + domain_type
    domain_taxonomy_by_engine.csv      per-engine, citation-weighted category %
    domain_taxonomy_audit.json         full audit trail
    _llm_classification_cache.json     PLD -> subcategory cache (reproducibility)

Run:
    python scripts/build_domain_taxonomy.py [--write] [--skip-llm]
      --write     overwrite the output artefacts (otherwise dry-run)
      --skip-llm  apply cached LLM classifications only; make no API calls
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import tldextract

try:  # only needed for PHASE 2 with uncached PLDs; --skip-llm runs without it
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

ROOT = Path(__file__).resolve().parent.parent
SILVER = ROOT / "data" / "scaleup" / "silver"
OUTDIR = ROOT / "data" / "scaleup" / "enrichment"
OUTDIR.mkdir(parents=True, exist_ok=True)

# tldextract with a cached PSL snapshot in the repo's cache dir (no network at
# classify time once warmed). suffix_list_urls=None forces the bundled snapshot.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=None)

# ---------------------------------------------------------------------------
# Curated lists. Match is by PAY-LEVEL DOMAIN (pld) unless noted *_FULL, which
# match the full hostname (for special subdomains whose pld belongs to another
# category, e.g. scholar.google.com).
# ---------------------------------------------------------------------------

ACADEMIC_PLDS = {
    "nih.gov", "ncbi.nlm.nih.gov", "arxiv.org", "biorxiv.org", "medrxiv.org",
    "sciencedirect.com", "springer.com", "springeropen.com", "nature.com",
    "wiley.com", "onlinelibrary.wiley.com", "jstor.org", "researchgate.net",
    "semanticscholar.org", "plos.org", "frontiersin.org", "mdpi.com",
    "tandfonline.com", "sagepub.com", "cell.com", "pnas.org", "bmj.com",
    "thelancet.com", "acs.org", "aps.org", "ieee.org", "acm.org",
    "elsevier.com", "academic.oup.com", "oup.com", "cambridge.org",
    "journals.plos.org", "europepmc.org", "ssrn.com", "researchsquare.com",
    "ebsco.com",
}
# Full-hostname academic overrides: pld belongs to another category but this
# specific subdomain is a research resource.
ACADEMIC_FULL = {
    "scholar.google.com",   # pld google.com (commercial)
    "eric.ed.gov",          # pld ed.gov (government) — ERIC education research db
}

NEWS_DOMAINS = {
    # Global / US
    "nytimes.com", "washingtonpost.com", "wsj.com", "reuters.com",
    "apnews.com", "bbc.com", "bbc.co.uk", "cnn.com", "cnbc.com",
    "nbcnews.com", "abcnews.go.com", "cbsnews.com", "foxnews.com",
    "theguardian.com", "ft.com", "bloomberg.com", "economist.com",
    "time.com", "newsweek.com", "usatoday.com", "latimes.com",
    "politico.com", "thehill.com", "axios.com", "vox.com",
    "huffpost.com", "buzzfeednews.com", "npr.org", "pbs.org",
    "aljazeera.com", "france24.com", "dw.com", "espn.com",
    "theconversation.com", "pcmag.com", "tomshardware.com", "space.com",
    # Tech news
    "techcrunch.com", "theverge.com", "wired.com", "arstechnica.com",
    "zdnet.com", "cnet.com", "engadget.com", "thenextweb.com",
    "venturebeat.com", "mashable.com", "gizmodo.com",
    # Science / health news
    "scientificamerican.com", "livescience.com", "sciencedaily.com",
    "newscientist.com", "sciencenews.org",
    # Business news
    "forbes.com", "businessinsider.com", "fortune.com",
    "inc.com", "fastcompany.com", "hbr.org",
    # Irish / UK
    "irishtimes.com", "rte.ie", "independent.ie", "thejournal.ie",
    "telegraph.co.uk", "independent.co.uk", "mirror.co.uk",
    "dailymail.co.uk", "sky.com",
}

REFERENCE_DOMAINS = {
    "britannica.com", "investopedia.com", "merriam-webster.com",
    "dictionary.com", "webmd.com", "mayoclinic.org", "healthline.com",
    "statista.com", "khanacademy.org", "howstuffworks.com",
    "clevelandclinic.org", "medicalnewstoday.com", "vocabulary.com",
    "collinsdictionary.com", "grokipedia.com", "worldhistory.org",
    "archive.org", "gutenberg.org", "thesaurus.com",
}
# Full-hostname reference overrides (checked before academic so a dictionary
# subdomain of a university press resolves to reference, not academic).
REFERENCE_FULL = {
    "dictionary.cambridge.org",   # pld cambridge.org (academic press)
    "baike.baidu.com",            # pld baidu.com (search) — Baidu Encyclopedia
}

UGC_DOMAINS = {
    "reddit.com", "quora.com", "stackoverflow.com", "stackexchange.com",
    "medium.com", "substack.com", "github.com", "gitlab.com",
    "linkedin.com", "twitter.com", "x.com", "youtube.com",
    "facebook.com", "tiktok.com", "fandom.com", "wordpress.com",
    "scribd.com", "blogspot.com", "tumblr.com", "pinterest.com",
    "tripadvisor.com",
}
# Full-hostname UGC overrides (pld belongs to another category).
UGC_FULL = {
    "zhidao.baidu.com",   # pld baidu.com (search) — Baidu Knows Q&A
}

# Government pay-level domains that lack a government suffix (caught by pld set).
# Includes national public-health services on a non-gov suffix (NHS) and
# intergovernmental organisations (UN family, OECD, IMF, World Bank, WTO, IEA,
# IAEA). .int treaty orgs (who.int, esa.int, reliefweb.int) and gov suffixes are
# handled separately in is_government_domain().
GOV_PLDS = {
    "canada.ca", "europa.eu", "admin.ch", "nhs.uk",
    "un.org", "oecd.org", "iea.org", "iaea.org", "imf.org",
    "worldbank.org", "wto.org", "unesco.org", "unicef.org",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_pay_level_domain(domain: str) -> str:
    """Registrable (pay-level) domain via the public-suffix list.
    en.wikipedia.org -> wikipedia.org ; docs.github.com -> github.com ;
    math.stackexchange.com -> stackexchange.com ; service.gov.uk -> service.gov.uk
    """
    ext = _EXTRACT(domain)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    # no recognised suffix (e.g. '**', bare IP) — return cleaned input
    return domain.lower().strip()


def _suffix(domain: str) -> str:
    return _EXTRACT(domain).suffix.lower()


def is_academic_suffix(suffix: str) -> bool:
    return suffix == "edu" or suffix.startswith("edu.") or suffix.startswith("ac.")


def in_full_set(domain: str, full_set: set) -> bool:
    """True if `domain` is, or is a subdomain of, any host in `full_set`.
    eric.ed.gov and files.eric.ed.gov both match the override 'eric.ed.gov'."""
    return any(domain == e or domain.endswith("." + e) for e in full_set)


def matches_set(domain: str, pld: str, curated: set) -> bool:
    """Robust curated-list membership. Normally the pay-level domain is the key,
    but tldextract's PSL lists some registrable domains (nhs.uk, wordpress.com,
    blogspot.com) as public suffixes themselves, so www./<user>. subdomains get a
    pld of www.nhs.uk etc. The subdomain (in_full_set) check catches those."""
    return pld in curated or in_full_set(domain, curated)


def is_government_domain(domain: str) -> bool:
    suffix = _suffix(domain)
    if suffix in {"gov", "mil", "int"}:
        return True
    if suffix.startswith(("gov.", "gob.", "gouv.", "go.", "gc.", "mil.")):
        return True
    if matches_set(domain, get_pay_level_domain(domain), GOV_PLDS):
        return True
    # string-ending fallback for suffixes the PSL snapshot might miss
    if domain.lower().endswith((".gov", ".mil", ".gc.ca")):
        return True
    return False


def classify_domain(domain: str) -> str:
    """Classify a domain into the taxonomy. Order matters — first match wins."""
    d = str(domain).lower().strip()
    pld = get_pay_level_domain(d)

    # Layer 1: Wikipedia / Wikimedia
    if any(x in d for x in ("wikipedia.org", "wikimedia.org", "wikidata.org", "wiktionary.org")):
        return "wikipedia"

    # Layer 1b: explicit full-hostname reference overrides (before academic)
    if in_full_set(d, REFERENCE_FULL):
        return "reference"

    # Layer 2: Academic (before government: nih.gov etc. are .gov but academic)
    if is_academic_suffix(_suffix(d)):
        return "academic"
    if matches_set(d, pld, ACADEMIC_PLDS) or in_full_set(d, ACADEMIC_FULL):
        return "academic"

    # Layer 3: Government
    if is_government_domain(d):
        return "government"

    # Layer 4: News / Media
    if matches_set(d, pld, NEWS_DOMAINS):
        return "news_media"

    # Layer 5: Reference / Knowledge
    if matches_set(d, pld, REFERENCE_DOMAINS):
        return "reference"

    # Layer 6: UGC / Forum
    if matches_set(d, pld, UGC_DOMAINS) or in_full_set(d, UGC_FULL):
        return "ugc_forum"

    # Layer 7: Residual
    return "commercial"


# ===========================================================================
# PHASE 2 — LLM granular industry labelling (Claude Haiku 4.5)
# ---------------------------------------------------------------------------
# The six rule-based categories above are never touched. Only domains labelled
# `commercial` (the rule-based residual) are sent to the LLM, which assigns each
# a single-word industry label from the predefined set below — granular enough
# for an SEO/GEO manager to know what kind of competitor occupies a citation
# slot. Deterministic (temperature 0) and disk-cached for reproducibility.
# ===========================================================================

MODEL = "claude-haiku-4-5-20251001"  # NOT one of the five engines under study
TEMPERATURE = 0                       # deterministic
MAX_TOKENS = 8192
BATCH_SIZE = 50                       # PLDs per API call
CACHE_FILE = OUTDIR / "_llm_classification_cache.json"

RULE_BASED_CATS = [
    "wikipedia", "academic", "government", "news_media", "reference", "ugc_forum",
]

INDUSTRIES = {
    # --- Technology ---
    "cybersecurity": (
        "Security companies, antivirus, VPN providers, identity management, "
        "encryption, threat intelligence, penetration testing. "
        "Examples: crowdstrike.com, norton.com, nordvpn.com, okta.com"
    ),
    "cloud": (
        "Cloud infrastructure, IaaS/PaaS platforms, serverless, containers, "
        "data centres, CDN providers, cloud storage. "
        "Examples: aws.amazon.com -> amazon.com counts here only if primary use is cloud, "
        "digitalocean.com, cloudflare.com, akamai.com, linode.com"
    ),
    "saas": (
        "SaaS productivity tools, CRM, project management, collaboration, "
        "marketing automation, communication platforms, HR software. "
        "Examples: salesforce.com, slack.com, asana.com, hubspot.com, mailchimp.com, "
        "notion.so, canva.com, zoom.us"
    ),
    "ai": (
        "AI and machine learning companies, generative AI tools, LLM providers, "
        "computer vision, NLP platforms, AI research labs, AI-powered products. "
        "Examples: openai.com, huggingface.co, midjourney.com, stability.ai, "
        "jasper.ai, copy.ai"
    ),
    "devtools": (
        "Developer tools, IDEs, code hosting beyond github/gitlab (already UGC), "
        "CI/CD, package managers, API platforms, documentation tools, testing frameworks. "
        "Examples: jetbrains.com, postman.com, vercel.com, netlify.com, "
        "hashicorp.com, docker.com, sentry.io"
    ),
    "analytics": (
        "Analytics platforms, business intelligence, SEO/SEM tools, data visualisation, "
        "A/B testing, web analytics, market research tools. "
        "Examples: semrush.com, ahrefs.com, mixpanel.com, tableau.com, "
        "hotjar.com, similarweb.com, moz.com"
    ),
    "hardware": (
        "Hardware manufacturers, consumer electronics, semiconductors, computer "
        "components, peripherals, networking equipment, IoT devices. "
        "Examples: intel.com, nvidia.com, apple.com (if hardware-primary), "
        "samsung.com, logitech.com, dell.com, lenovo.com"
    ),
    # --- Finance & Professional Services ---
    "fintech": (
        "Payment processors, digital banking, neobanks, crypto exchanges, "
        "blockchain platforms, mobile payments, lending platforms. "
        "Examples: stripe.com, coinbase.com, revolut.com, square.com, robinhood.com"
    ),
    "banking": (
        "Traditional banks, credit unions, investment banks, wealth management, "
        "brokerage firms, stock exchanges. "
        "Examples: jpmorgan.com, goldmansachs.com, vanguard.com, fidelity.com"
    ),
    "insurance": (
        "Insurance companies, insurtech, benefits platforms, actuarial services. "
        "Examples: geico.com, lemonade.com, progressive.com, aetna.com"
    ),
    "consulting": (
        "Management consulting, strategy firms, advisory, professional services, "
        "accounting firms, tax services, audit firms. "
        "Examples: mckinsey.com, deloitte.com, pwc.com, accenture.com, kpmg.com"
    ),
    "legal": (
        "Law firms, legal technology, legal information services, court systems, "
        "contract management, compliance platforms. "
        "Examples: legalzoom.com, westlaw.com, findlaw.com, clio.com"
    ),
    "realestate": (
        "Real estate platforms, property listings, mortgage lenders, property "
        "management, home valuation, rental platforms. "
        "Examples: zillow.com, realtor.com, redfin.com, apartments.com, airbnb.com"
    ),
    # --- Commerce ---
    "ecommerce": (
        "Online marketplaces, general retailers, auction sites, price comparison, "
        "coupon and deal sites, dropshipping platforms. "
        "Examples: amazon.com, ebay.com, walmart.com, target.com, aliexpress.com"
    ),
    "fashion": (
        "Fashion brands, apparel retailers, luxury goods, accessories, shoes, "
        "beauty and cosmetics, personal care. "
        "Examples: nike.com, zara.com, sephora.com, nordstrom.com, glossier.com"
    ),
    # --- Health ---
    "healthcare": (
        "Hospitals, clinics, health systems, telemedicine, mental health platforms, "
        "dental, vision, veterinary services, elder care. "
        "Examples: clevelandclinic.org, teladoc.com, zocdoc.com, betterhelp.com"
    ),
    "pharma": (
        "Pharmaceutical companies, biotech, drug manufacturers, medical devices, "
        "laboratory supply, clinical research, life sciences. "
        "Examples: pfizer.com, thermofisher.com, merck.com, abbvie.com, roche.com"
    ),
    "fitness": (
        "Fitness brands, gym chains, sports equipment, athletic wear, "
        "wellness apps, nutrition supplements, wearable fitness tech. "
        "Examples: peloton.com, myfitnesspal.com, underarmour.com, garmin.com"
    ),
    # --- Education ---
    "edtech": (
        "Online learning platforms, course marketplaces, educational technology, "
        "language learning, test prep, certification, tutoring, coding bootcamps. "
        "Examples: coursera.org, udemy.com, duolingo.com, codecademy.com, "
        "skillshare.com, pluralsight.com, brilliant.org"
    ),
    # --- Media & Entertainment ---
    "streaming": (
        "Video streaming, music streaming, podcast platforms, audiobook services, "
        "live streaming, content delivery for media. "
        "Examples: netflix.com, spotify.com, hulu.com, twitch.tv, audible.com"
    ),
    "gaming": (
        "Video game companies, gaming platforms, esports, game development tools, "
        "game review sites, mobile gaming. "
        "Examples: steam.com, epicgames.com, riotgames.com, ign.com, ea.com"
    ),
    "travel": (
        "Travel booking, airlines, hotels, car rental, cruise lines, travel guides, "
        "tourism boards, vacation rental platforms. "
        "Examples: booking.com, expedia.com, tripadvisor.com, marriott.com, "
        "kayak.com, lonelyplanet.com"
    ),
    "food": (
        "Food and recipe sites, restaurant chains, food delivery, meal kits, "
        "grocery delivery, food review, cooking tools and appliances. "
        "Examples: allrecipes.com, doordash.com, hellofresh.com, dominos.com, "
        "seriouseats.com, instacart.com"
    ),
    "sports": (
        "Sports leagues, teams, sporting goods retailers, sports media, "
        "fantasy sports, sports betting, athletic events. "
        "Examples: espn.com (if not news_media), nba.com, nfl.com, draftkings.com"
    ),
    # --- Infrastructure & Industry ---
    "telecom": (
        "Telecommunications, mobile carriers, ISPs, cable companies, "
        "unified communications, VoIP providers. "
        "Examples: verizon.com, att.com, t-mobile.com, comcast.com, vonage.com"
    ),
    "automotive": (
        "Car manufacturers, auto dealers, auto parts, car review, "
        "electric vehicles, ride-sharing, fleet management. "
        "Examples: tesla.com, ford.com, carvana.com, autotrader.com, uber.com, lyft.com"
    ),
    "energy": (
        "Energy companies, utilities, oil and gas, renewable energy, solar, "
        "electric utilities, energy technology. "
        "Examples: nexteraenergy.com, enphase.com, shell.com"
    ),
    "manufacturing": (
        "Manufacturing companies, industrial equipment, construction, "
        "building materials, industrial supply, logistics, shipping. "
        "Examples: caterpillar.com, 3m.com, grainger.com, fedex.com, ups.com"
    ),
    # --- Civic & Nonprofit ---
    "nonprofit": (
        "Charities, foundations, humanitarian organisations, advocacy groups, "
        "environmental organisations, social enterprises, religious organisations. "
        "Examples: redcross.org, greenpeace.org, unicef.org, habitat.org"
    ),
    "standards": (
        "Standards bodies, professional associations, industry consortia, "
        "open-source foundations, technical standards organisations. "
        "Examples: w3.org, ieee.org, iso.org, apache.org, linux.org, ietf.org"
    ),
    # --- Residual ---
    "other": (
        "Any domain that does not clearly fit the categories above. "
        "Small businesses, niche industry sites, personal brands, portfolios, "
        "local businesses, parked domains, unrecognisable sites."
    ),
}
VALID_LABELS = set(INDUSTRIES.keys())

SYSTEM_PROMPT = """You are a domain classifier that assigns a single industry label to each website domain.

INDUSTRIES (pick exactly one per domain):
{industries}

RULES:
1. Classify by the PRIMARY business of the organisation owning the domain.
2. One word only — use exactly the label from the list above.
3. If a domain fits multiple labels, pick the one describing its core revenue source.
4. If you do not recognise the domain, classify as "other".
5. Respond with ONLY a JSON object mapping each domain to its label. No markdown, no explanation.

Example input: ["stripe.com", "crowdstrike.com", "zillow.com", "w3.org", "randomsite.xyz"]
Example output: {{"stripe.com": "fintech", "crowdstrike.com": "cybersecurity", "zillow.com": "realestate", "w3.org": "standards", "randomsite.xyz": "other"}}"""


def build_system_prompt() -> str:
    lines = "\n".join(f"- {label}: {desc}" for label, desc in INDUSTRIES.items())
    return SYSTEM_PROMPT.format(industries=lines)


# ===========================================================================
# PHASE 3a — rule-based sub-classification of `academic`
#   academic -> journal | preprint | database | university | research
# ===========================================================================

JOURNAL_PUBLISHERS = {
    "nature.com", "springer.com", "wiley.com", "elsevier.com",
    "sciencedirect.com", "tandfonline.com", "sagepub.com", "mdpi.com",
    "frontiersin.org", "cell.com", "thelancet.com", "bmj.com",
    "plos.org", "pnas.org", "oup.com", "academic.oup.com",
    "annualreviews.org", "cambridge.org", "degruyter.com",
    "acs.org", "rsc.org", "iop.org", "aip.org",
    "nejm.org", "jama.com", "jamanetwork.com",
    "karger.com", "hindawi.com", "iospress.com",
    "liebertpub.com", "futuremed.com", "thieme.com",
}
PREPRINT_SERVERS = {
    "arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com",
    "preprints.org", "osf.io", "zenodo.org", "figshare.com",
    "researchsquare.com", "techrxiv.org", "eartharxiv.org",
    "chemrxiv.org", "engrxiv.org", "psyarxiv.com",
}
RESEARCH_DATABASES = {
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "scholar.google.com", "semanticscholar.org",
    "jstor.org", "researchgate.net", "dimensions.ai",
    "webofscience.com", "scopus.com", "crossref.org",
    "doaj.org", "core.ac.uk", "base-search.net",
    "europepmc.org", "lens.org",
}
ACADEMIC_SUBLABELS = ["journal", "preprint", "database", "university", "research"]
_EDU_TLDS = (".edu", ".ac.uk", ".ac.jp", ".edu.au", ".edu.sg", ".ac.in",
             ".edu.cn", ".ac.nz", ".edu.br", ".ac.za", ".edu.mx",
             ".ac.kr", ".edu.tr", ".edu.pl", ".uni-", ".university")


def classify_academic(domain: str, pld: str) -> str:
    d, p = domain.lower(), pld.lower()
    if p in PREPRINT_SERVERS or any(x in d for x in PREPRINT_SERVERS):
        return "preprint"
    if p in RESEARCH_DATABASES or any(x in d for x in RESEARCH_DATABASES):
        return "database"
    if p in JOURNAL_PUBLISHERS or any(x in d for x in JOURNAL_PUBLISHERS):
        return "journal"
    if any(p.endswith(t) or t in d for t in _EDU_TLDS):
        return "university"
    return "research"  # research institutes, labs, academic-character residual


# ===========================================================================
# PHASE 3b — rule-based sub-classification of `ugc_forum`
#   ugc_forum -> forum | code | video | blogging | social | community
# ===========================================================================

FORUM_QA = {
    "reddit.com", "quora.com", "stackoverflow.com", "stackexchange.com",
    "discourse.org", "producthunt.com", "slashdot.org",
    "hackernews.com", "news.ycombinator.com",
    "answers.com", "superuser.com", "serverfault.com",
    "askubuntu.com", "mathoverflow.net",
}
SOCIAL_MEDIA = {
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "linkedin.com", "tiktok.com", "pinterest.com",
    "threads.net", "mastodon.social", "bsky.app",
    "tumblr.com", "snapchat.com",
}
VIDEO_PLATFORMS = {
    "youtube.com", "vimeo.com", "dailymotion.com", "rumble.com", "bitchute.com",
}
BLOGGING_PLATFORMS = {
    "medium.com", "substack.com", "wordpress.com",
    "blogger.com", "blogspot.com", "ghost.org",
    "hashnode.dev", "dev.to", "mirror.xyz",
    "wix.com", "squarespace.com", "weebly.com",
}
CODE_PLATFORMS = {
    "github.com", "gitlab.com", "bitbucket.org",
    "codepen.io", "replit.com", "codesandbox.io",
    "kaggle.com", "huggingface.co", "observablehq.com",
}
UGC_SUBLABELS = ["forum", "code", "video", "blogging", "social", "community"]


def classify_ugc(domain: str, pld: str) -> str:
    d, p = domain.lower(), pld.lower()
    if "stackexchange.com" in d or p in FORUM_QA:
        return "forum"
    if p in CODE_PLATFORMS:
        return "code"
    if p in VIDEO_PLATFORMS:
        return "video"
    if p in BLOGGING_PLATFORMS:
        return "blogging"
    if p in SOCIAL_MEDIA:
        return "social"
    return "community"  # community sites not matching a specific platform


# ===========================================================================
# PHASE 3c — LLM re-classification of the `other` residual (Claude Haiku 4.5)
# ---------------------------------------------------------------------------
# Additional long-tail labels for domains Phase 2 could not place in an
# industry. NOTE: to honour the conservation constraint (the 25 industry
# categories must keep identical counts), Phase 3c offers ONLY these new labels
# plus `other` — it does NOT reassign residual domains into existing industries.
# ===========================================================================

ADDITIONAL_LABELS = {
    "blog": (
        "Self-hosted blogs, personal websites, opinion sites, niche content sites "
        "run by individuals or small teams — NOT platform blogs like Medium/Substack. "
        "Examples: paulgraham.com, waitbutwhy.com, stratechery.com"
    ),
    "directory": (
        "Business directories, listing sites, review aggregators, comparison sites, "
        "rating platforms, yellow pages, classifieds. "
        "Examples: yelp.com, glassdoor.com, trustpilot.com, g2.com, capterra.com, "
        "yellowpages.com, craigslist.org, angi.com"
    ),
    "marketplace": (
        "Freelance marketplaces, gig platforms, service marketplaces, talent platforms, "
        "creative asset marketplaces. "
        "Examples: fiverr.com, upwork.com, thumbtack.com, envato.com, shutterstock.com, "
        "gettyimages.com"
    ),
    "utility": (
        "Small utility sites, online calculators, converters, generators, "
        "URL shorteners, file converters, productivity micro-tools. "
        "Examples: convertio.co, smallpdf.com, tinyurl.com, remove.bg, "
        "regex101.com, jsonlint.com"
    ),
    "retail": (
        "Small or niche retailers, direct-to-consumer brands, specialty shops, "
        "product-specific stores — distinct from large ecommerce marketplaces. "
        "Examples: warbyparker.com, allbirds.com, glossier.com, casper.com"
    ),
    "agency": (
        "Marketing agencies, design agencies, web development shops, PR firms, "
        "recruitment agencies, staffing companies, ad agencies. "
        "Examples: wpp.com, ogilvy.com, 99designs.com, toptal.com"
    ),
    "government_services": (
        "Government service portals, public records, tax filing, permit systems, "
        "public health services, civil registries — sites that deliver government "
        "services but are not on .gov TLDs. "
        "Examples: turbotax.com (tax), dmv.org (unofficial)"
    ),
}
# valid OUTPUTS for Phase 3c (industries deliberately excluded — see note above)
VALID_RECLASSIFY = set(ADDITIONAL_LABELS) | {"other"}
# every label the cache may legitimately hold (Phase 2 + Phase 3c)
ALL_LLM_LABELS = set(INDUSTRIES) | set(ADDITIONAL_LABELS)

SYSTEM_PROMPT_OTHER = """You are a domain classifier. Each domain gets exactly one single-word label.

IMPORTANT: These domains were previously classified as "other" because they did not fit standard industry categories. Try harder this time. Most domains DO have a recognisable purpose — look at the domain name carefully. Only use "other" if the domain is truly unrecognisable (parked, defunct, or impossibly niche).

LABELS:
{labels}

RULES:
1. Classify by the PRIMARY purpose of the domain owner.
2. Pick exactly one label from the list above.
3. If genuinely unrecognisable, use "other" — but this should be rare.
4. Respond with ONLY a JSON object. No markdown, no explanation.

Example input: ["paulgraham.com", "yelp.com", "fiverr.com", "remove.bg", "xyzrandom123.com"]
Example output: {{"paulgraham.com": "blog", "yelp.com": "directory", "fiverr.com": "marketplace", "remove.bg": "utility", "xyzrandom123.com": "other"}}"""


def build_other_prompt() -> str:
    lines = "\n".join(f"- {label}: {desc}" for label, desc in ADDITIONAL_LABELS.items())
    return SYSTEM_PROMPT_OTHER.format(labels=lines)


def _load_dotenv() -> None:
    """Populate ANTHROPIC_API_KEY from the repo .env if not already in env.
    Never prints the secret."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_cache() -> dict:
    """Load the disk cache, keeping only entries whose value is a label from the
    CURRENT scheme (Phase 2 industries or Phase 3c additional labels). Entries
    written under an older scheme are dropped so those PLDs are re-classified."""
    if not CACHE_FILE.exists():
        return {}
    old = json.loads(CACHE_FILE.read_text())
    cache = {k: v for k, v in old.items() if v in ALL_LLM_LABELS}
    print(f"Loaded cache: {len(cache):,} usable PLD labels (of {len(old):,} on disk)")
    return cache


def _classify_batch(client, domains: list, system: str, valid_labels: set) -> dict:
    """Classify one batch via Claude Haiku, validating against `valid_labels`."""
    user_msg = json.dumps(domains)
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):  # strip markdown fences if present
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            for d in domains:  # every input must be present with a valid label
                if d not in result or result[d] not in valid_labels:
                    if d in result:
                        print(f"  WARNING: invalid '{result[d]}' for '{d}' -> other")
                    result[d] = "other"
            return {d: result[d] for d in domains}
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  Attempt {attempt + 1}/3 failed ({type(e).__name__}: {e}), retrying...")
            time.sleep(2 ** attempt)
        except anthropic.RateLimitError:
            print("  Rate limited, waiting 30s...")
            time.sleep(30)
        except anthropic.APIError as e:
            print(f"  API error: {e}, retrying...")
            time.sleep(5)
    print(f"  ALL RETRIES FAILED for batch of {len(domains)}, defaulting to other")
    return {d: "other" for d in domains}


def run_llm_pass(plds_todo: list, system: str, valid_labels: set,
                 cache: dict, skip_llm: bool, tag: str) -> None:
    """Classify `plds_todo` into `cache` in place (batched, disk-cached after each
    batch). No-op under --skip-llm. `plds_todo` is pre-filtered by the caller to
    exactly the PLDs that still need a label in this pass."""
    print(f"[{tag}] PLDs needing classification: {len(plds_todo):,}")
    if not plds_todo:
        return
    if skip_llm:
        print(f"[--skip-llm] {len(plds_todo):,} uncached PLDs default to 'other' (no API calls)")
        return
    if anthropic is None:
        raise RuntimeError("anthropic SDK not installed — run `pip install anthropic` "
                           "or pass --skip-llm to use the cache only")
    _load_dotenv()
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
    batches = [plds_todo[i:i + BATCH_SIZE] for i in range(0, len(plds_todo), BATCH_SIZE)]
    print(f"[{tag}] {len(batches)} batches of up to {BATCH_SIZE} | model={MODEL} temp={TEMPERATURE}")
    for i, batch in enumerate(batches):
        print(f"  [{tag}] batch {i + 1}/{len(batches)} ({len(batch)})...", end=" ", flush=True)
        cache.update(_classify_batch(client, batch, system, valid_labels))
        print(f"done  [cache: {len(cache):,}]")
        CACHE_FILE.write_text(json.dumps(cache))  # crash-safe resume after each batch
        if i < len(batches) - 1:
            time.sleep(0.5)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))  # final, pretty-printed


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def main(write: bool, skip_llm: bool) -> None:
    citations = pd.read_parquet(SILVER / "citations_scaleup.parquet")
    print(f"Citations: {len(citations):,} rows, {citations['domain'].nunique():,} unique domains")

    # ---------------- PHASE 1: rule-based classification ----------------
    uniq = pd.Series(citations["domain"].dropna().unique(), name="domain")
    taxonomy = pd.DataFrame({"domain": uniq})
    taxonomy["pay_level_domain"] = taxonomy["domain"].map(get_pay_level_domain)
    taxonomy["domain_type"] = taxonomy["domain"].map(classify_domain)

    weighted = citations.merge(taxonomy[["domain", "domain_type"]], on="domain", how="left")
    print("\n=== PHASE 1 — RULE-BASED DISTRIBUTION (citation-weighted %) ===")
    print(weighted["domain_type"].value_counts(normalize=True, dropna=True).mul(100).round(1).to_string())

    # ---------------- PHASE 2: LLM industry labels on the `commercial` residual ------
    print("\n" + "=" * 60)
    print("PHASE 2 — LLM INDUSTRY LABELS (commercial residual)")
    print("=" * 60)
    cache = load_cache()
    commercial_plds = taxonomy.loc[taxonomy.domain_type == "commercial", "pay_level_domain"].unique().tolist()
    print(f"Commercial domains: {int((taxonomy.domain_type == 'commercial').sum()):,} | "
          f"unique PLDs: {len(commercial_plds):,}")
    p2_todo = [p for p in commercial_plds if p not in cache]
    run_llm_pass(p2_todo, build_system_prompt(), VALID_LABELS, cache, skip_llm, "P2-industry")

    cmask = taxonomy.domain_type == "commercial"
    taxonomy.loc[cmask, "domain_type"] = taxonomy.loc[cmask, "pay_level_domain"].map(lambda p: cache.get(p, "other"))
    assert (taxonomy.domain_type == "commercial").sum() == 0, "'commercial' residual not fully labelled"

    # conservation baseline: EVERY category except the three being split (academic,
    # ugc_forum, other) must keep identical citation counts through Phase 3.
    weighted = citations.merge(taxonomy[["domain", "domain_type"]], on="domain", how="left")
    SPLIT_CATS = ["academic", "ugc_forum", "other"]
    untouched_snapshot = weighted[~weighted.domain_type.isin(SPLIT_CATS)]["domain_type"].value_counts()

    # ---------------- PHASE 3: sub-classify academic / ugc_forum / other ----------------
    print("\n" + "=" * 60)
    print("PHASE 3 — SUB-CLASSIFY academic / ugc_forum / other")
    print("=" * 60)

    # 3a — academic (rule-based)
    amask = taxonomy.domain_type == "academic"
    n_acad = int(amask.sum())
    taxonomy.loc[amask, "domain_type"] = taxonomy.loc[amask].apply(
        lambda r: classify_academic(r["domain"], r["pay_level_domain"]), axis=1)
    print(f"[3a] academic ({n_acad:,} domains) -> "
          f"{taxonomy.loc[taxonomy.domain_type.isin(ACADEMIC_SUBLABELS), 'domain_type'].value_counts().to_dict()}")

    # 3b — ugc_forum (rule-based)
    umask = taxonomy.domain_type == "ugc_forum"
    n_ugc = int(umask.sum())
    taxonomy.loc[umask, "domain_type"] = taxonomy.loc[umask].apply(
        lambda r: classify_ugc(r["domain"], r["pay_level_domain"]), axis=1)
    print(f"[3b] ugc_forum ({n_ugc:,} domains) -> "
          f"{taxonomy.loc[taxonomy.domain_type.isin(UGC_SUBLABELS), 'domain_type'].value_counts().to_dict()}")

    # 3c — other (LLM, additional long-tail labels only; industries NOT reassignable)
    omask = taxonomy.domain_type == "other"
    other_plds = taxonomy.loc[omask, "pay_level_domain"].unique().tolist()
    print(f"[3c] 'other' residual: {int(omask.sum()):,} domains | unique PLDs: {len(other_plds):,}")
    p3c_todo = [p for p in other_plds if cache.get(p, "other") == "other"]
    run_llm_pass(p3c_todo, build_other_prompt(), VALID_RECLASSIFY, cache, skip_llm, "P3c-other")
    taxonomy.loc[omask, "domain_type"] = taxonomy.loc[omask, "pay_level_domain"].map(lambda p: cache.get(p, "other"))

    weighted = citations.merge(taxonomy[["domain", "domain_type"]], on="domain", how="left")

    # ===================== CROSS-AUDIT =====================
    NEW_SUBLABELS = ACADEMIC_SUBLABELS + UGC_SUBLABELS + list(ADDITIONAL_LABELS)
    order = (["wikipedia", "government", "news_media", "reference"]
             + [c for c in INDUSTRIES if c != "other"]
             + NEW_SUBLABELS + ["other"])
    print("\n" + "=" * 60)
    print("CROSS-AUDIT")
    print("=" * 60)

    print("\n=== CHECK 1 — FULL CITATION-WEIGHTED DISTRIBUTION (%) ===")
    print(weighted["domain_type"].value_counts(normalize=True, dropna=True).mul(100).round(1).to_string())
    print(f"Total categories: {taxonomy.domain_type.nunique()}")
    assert len(taxonomy) == 10155, f"TAXONOMY ROW COUNT CHANGED: {len(taxonomy):,}"

    print("\n=== CHECK 2 — CONSERVATION OF UNTOUCHED CATEGORIES ===")
    new_untouched = weighted[~weighted.domain_type.isin(SPLIT_CATS)]["domain_type"].value_counts()
    mismatches = []
    for cat in untouched_snapshot.index:
        old, new = int(untouched_snapshot[cat]), int(new_untouched.get(cat, 0))
        if old != new:
            mismatches.append(f"  {cat}: {old:,} -> {new:,} MISMATCH")
    if mismatches:
        print("\n".join(mismatches))
    assert not mismatches, "UNTOUCHED CATEGORIES WERE MODIFIED — critical error"
    print(f"All {len(untouched_snapshot)} untouched categories conserved (4 structural + "
          f"{len(untouched_snapshot) - 4} industries)")

    print("\n=== CHECK 3 — OLD LABELS ELIMINATED ===")
    for lab in ("academic", "ugc_forum"):
        assert lab not in taxonomy.domain_type.values, f"'{lab}' still present"
    other_c = int((weighted.domain_type == "other").sum())
    print(f"'academic' and 'ugc_forum' fully replaced | residual 'other': "
          f"{int((taxonomy.domain_type == 'other').sum()):,} domains, "
          f"{other_c:,} citations ({other_c/len(weighted)*100:.1f}%)")

    print("\n=== CHECK 4 — TOP 10 DOMAINS PER NEW SUBLABEL ===")
    for label in NEW_SUBLABELS:
        doms = set(taxonomy.loc[taxonomy.domain_type == label, "domain"])
        sub = weighted[weighted["domain"].isin(doms)]
        if not len(sub):
            print(f"\n--- {label}: EMPTY ---")
            continue
        print(f"\n--- {label} ({len(doms):,} domains, {len(sub):,} citations, "
              f"{len(sub)/len(weighted)*100:.1f}%) ---")
        print(sub["domain"].value_counts().head(10).to_string())

    print("\n=== CHECK 5 — PER-ENGINE DISTRIBUTION (%) ===")
    cross = pd.crosstab(weighted["engine"], weighted["domain_type"], normalize="index").mul(100).round(1)
    cross = cross.reindex(columns=[c for c in order if c in cross.columns])
    print(cross.to_string())

    print("\n=== CHECK 6 — LOW-COUNT CATEGORIES (<1% of citations; retained, not merged) ===")
    full_dist = weighted["domain_type"].value_counts(normalize=True, dropna=True).mul(100).round(2)
    structural = {"wikipedia", "government", "news_media", "reference"}
    for label, pct in full_dist.items():
        if pct < 1.0 and label not in structural:
            n = int((weighted["domain_type"] == label).sum())
            print(f"  {label}: {pct}% ({n:,} citations)")

    print("\n=== CHECK 7 — NaN AUDIT ===")
    cit_nan = int(weighted["domain_type"].isna().sum())
    print(f"Taxonomy NaN: {int(taxonomy.domain_type.isna().sum())} | "
          f"Citations NaN: {cit_nan} ({cit_nan/len(weighted)*100:.3f}%)")

    print("\n=== CHECK 8 — MUTUAL EXCLUSIVITY ===")
    multi = int((taxonomy.groupby("domain").domain_type.nunique() > 1).sum())
    assert multi == 0, f"MUTUAL EXCLUSIVITY VIOLATED: {multi} domains have multiple categories"
    print(f"All {len(taxonomy):,} domains have exactly one category")

    print("\n=== CHECK 9 — CATEGORY VALIDITY ===")
    all_valid = (structural | set(INDUSTRIES) | set(ACADEMIC_SUBLABELS)
                 | set(UGC_SUBLABELS) | set(ADDITIONAL_LABELS))
    invalid = set(taxonomy.domain_type.dropna().unique()) - all_valid
    assert not invalid, f"INVALID CATEGORIES FOUND: {invalid}"
    print(f"All categories valid ({taxonomy.domain_type.nunique()} present)")

    if not write:
        print("\n[dry-run] not writing outputs (pass --write to overwrite artefacts).")
        return

    # ===================== WRITE (overwrite the four artefacts) =====================
    print("\n" + "=" * 60)
    print("WRITING FILES")
    print("=" * 60)
    tax_path = OUTDIR / "domain_taxonomy.parquet"
    taxonomy.to_parquet(tax_path, index=False)
    print(f"Overwritten: {tax_path.name} ({len(taxonomy):,} rows)")

    enriched = citations.merge(taxonomy[["domain", "domain_type"]], on="domain", how="left")
    enr_path = OUTDIR / "citations_scaleup_enriched.parquet"
    enriched.to_parquet(enr_path, index=False)
    print(f"Overwritten: {enr_path.name} ({len(enriched):,} rows)")

    summary = pd.crosstab(weighted["engine"], weighted["domain_type"], normalize="index").mul(100).round(1)
    summary = summary.reindex(columns=[c for c in order if c in summary.columns])
    sum_path = OUTDIR / "domain_taxonomy_by_engine.csv"
    summary.to_csv(sum_path)
    print(f"Overwritten: {sum_path.name}")

    all_labels = sorted(taxonomy.domain_type.dropna().unique())
    audit = {
        "classification_method": (
            "Three-phase hybrid: (1) rule-based structural categories -> (2) LLM industry labels "
            "for the commercial residual -> (3) rule-based sub-classification of academic and UGC, "
            "plus LLM re-classification of the 'other' residual with additional long-tail labels"
        ),
        "classified_on": "citations.domain (per-citation authoritative domain)",
        "llm_model": MODEL,
        "llm_model_rationale": (
            "Claude Haiku 4.5 selected because it is NOT one of the five engines under study "
            "(ChatGPT, Claude, Gemini, Kimi, Perplexity), minimising circularity risk."
        ),
        "llm_temperature": TEMPERATURE,
        "llm_batch_size": BATCH_SIZE,
        "llm_plds_industry": len(commercial_plds),
        "llm_plds_other_reclassified": len(other_plds),
        "total_unique_domains": int(len(taxonomy)),
        "total_citations": int(len(weighted)),
        "total_categories": len(all_labels),
        "total_citations_with_type": int(weighted["domain_type"].notna().sum()),
        "total_citations_nan_type": int(weighted["domain_type"].isna().sum()),
        "all_category_labels": all_labels,
        "phase_1_structural": ["wikipedia", "government", "news_media", "reference"],
        "phase_2_llm_industry": [c for c in INDUSTRIES if c != "other"],
        "phase_3a_academic_sublabels": ACADEMIC_SUBLABELS,
        "phase_3b_ugc_sublabels": UGC_SUBLABELS,
        "phase_3c_other_additional_labels": list(ADDITIONAL_LABELS.keys()),
        "phase_3c_note": (
            "The 'other' re-classification offers only the additional long-tail labels plus 'other'; "
            "existing industry categories are deliberately NOT reassignable, to keep their citation "
            "counts conserved (see Check 2)."
        ),
        "category_counts_by_domain": taxonomy.domain_type.value_counts().to_dict(),
        "category_counts_by_citation": weighted["domain_type"].value_counts(dropna=True).to_dict(),
        "category_pct_by_citation": weighted["domain_type"].value_counts(normalize=True, dropna=True).mul(100).round(2).to_dict(),
        "per_engine_distribution": summary.to_dict(),
        "reproducibility": {
            "cache_file": str(CACHE_FILE.relative_to(ROOT)),
            "rebuild_command": "python scripts/build_domain_taxonomy.py --skip-llm --write",
            "determinism": "temperature=0; results cached to disk; reproduce via --skip-llm --write",
            "methodology_sentence": (
                "Cited domains were classified in three phases: rule-based assignment of structural "
                "categories (Wikipedia, government, news media, reference); LLM-based industry "
                "classification of the commercial residual using Claude Haiku 4.5 (Anthropic, 2025), "
                "a model not included in the study's engine set, at temperature zero; and rule-based "
                "sub-classification of academic sources (journal, university, preprint, database, "
                "research) and user-generated-content platforms (forum, code, video, blogging, social, "
                "community), with the unclassified residual re-examined by the same model against "
                "additional long-tail labels. All LLM classifications were cached for deterministic "
                "reproduction without API access."
            ),
        },
    }
    aud_path = OUTDIR / "domain_taxonomy_audit.json"
    aud_path.write_text(json.dumps(audit, indent=2))
    print(f"Overwritten: {aud_path.name}")

    # ===================== FINAL VERIFICATION =====================
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)
    tax_check = pd.read_parquet(tax_path)
    cit_check = pd.read_parquet(enr_path)
    assert len(tax_check) == 10155, f"Taxonomy: {len(tax_check):,} != 10,155"
    assert len(cit_check) == 58851, f"Citations: {len(cit_check):,} != 58,851"
    legacy = {"commercial", "academic", "ugc_forum", "tech_software", "finance_business",
              "media_entertainment", "health_science", "education_training", "ecommerce_retail",
              "nonprofit_ngo", "government_adjacent", "commercial_other"}
    present_legacy = legacy & set(tax_check.domain_type.values)
    assert not present_legacy, f"Legacy labels still present: {present_legacy}"
    assert CACHE_FILE.exists(), "Cache file missing"
    other_pct = (cit_check.domain_type == "other").sum() / len(cit_check) * 100
    print(f"Taxonomy: {len(tax_check):,} rows across {tax_check.domain_type.nunique()} categories")
    print(f"Citations: {len(cit_check):,} rows | 'academic' and 'ugc_forum' gone")
    print(f"Residual 'other': {other_pct:.1f}%")
    print(f"Cache file: {len(json.loads(CACHE_FILE.read_text())):,} PLDs "
          f"({CACHE_FILE.stat().st_size / 1024:.0f} KB)")
    print("\n=== FINAL TOP 20 (%) ===")
    print(cit_check["domain_type"].value_counts(normalize=True).mul(100).round(1).head(20).to_string())
    print("\n✓ ALL CHECKS PASSED — academic/UGC/other sub-classification complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the domain-type taxonomy (rule-based + LLM).")
    parser.add_argument("--write", action="store_true", help="overwrite the output artefacts")
    parser.add_argument("--skip-llm", action="store_true",
                        help="apply cached LLM classifications only; make no API calls")
    args = parser.parse_args()
    main(write=args.write, skip_llm=args.skip_llm)
