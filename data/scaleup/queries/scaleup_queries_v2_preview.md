# Scale-up Query Registry v2 — Preview

**File:** `data/scaleup/queries/scaleup_queries_v2.parquet`  
**Generated:** 2026-06-06 15:11 UTC  
**Total rows:** 250 (100 pilot_carryover + 150 fresh_draw_20260606)  
**Intent distribution:** 200 informational (80.0%) / 25 navigational (10.0%) / 25 transactional (10.0%)  

Human sanity-check sample: 10 pilot_carryover + 10 fresh_draw = 20 queries.

## Pilot carryover (first 10 of 100)

| # | query_id | inclusion_reason | primary_intent | query_text | tags |
|---|---|---|---|---|---|
| 1 | `gb_1a77ad01ab570671` | pilot_carryover | informational | when does elena turn into a vampire in the tv series | informational, simple, arts and entertainment, non-technical, question, fact, re |
| 2 | `gb_314f7fd2ce040788` | pilot_carryover | informational | who played the oldest brother in 7th heaven | informational, simple, arts and entertainment, non-technical, question, fact, re |
| 3 | `gb_eaab9147ac75176b` | pilot_carryover | informational | which came first the walking dead comic or show | informational, simple, arts and entertainment, non-technical, question, fact, re |
| 4 | `gb_51f95e9b9f4f6183` | pilot_carryover | informational | who plays unis in she's the man | informational, arts and entertainment, non-technical, question, fact, simple |
| 5 | `gb_f3a09c028168dc20` | pilot_carryover | informational | where does the sound come from when you crack your knuckles | informational, health, non-technical, question, fact, simple |
| 6 | `gb_57e198d356ad745f` | pilot_carryover | navigational | the book of the thousand nights and one night volume v | navigational, books and literature, non-technical, command, research, simple |
| 7 | `gb_588d144b9e0f9869` | pilot_carryover | informational | who sings the christmas song all i want for christmas is you | informational, arts and entertainment, non-technical, question, fact, simple |
| 8 | `gb_17d6f89c7d898640` | pilot_carryover | informational | when did gaurdians of the galaxy 2 come out | informational, arts and entertainment, non-technical, question, fact, historical |
| 9 | `gb_a33bd88b6b1ff82a` | pilot_carryover | informational | when will the next episode of flash be aired | informational, arts and entertainment, non-technical, question, prediction, simp |
| 10 | `gb_a61c19b7ea5345d6` | pilot_carryover | navigational | only fools and horses del falls through the bar episode | navigational, simple, non-technical, arts and entertainment, research |

## Fresh draw (first 10 of 150)

| # | query_id | inclusion_reason | primary_intent | query_text | tags |
|---|---|---|---|---|---|
| 1 | `gb_f80c681bf981289b` | fresh_draw_20260606 | informational | methane bond | informational, simple, science, chemistry, technical, statement, research, non-s |
| 2 | `gb_ef35a47d75d00885` | fresh_draw_20260606 | informational | Should labor organizations be granted the power to strike? | intermediate, debate, jobs and education, non-technical, question, opinion, rese |
| 3 | `gb_f080994068fc3afa` | fresh_draw_20260606 | informational | Why does my cat kick its toys when playing with them? | informational, simple, non-technical, question, explanation, non-sensitive, pets |
| 4 | `gb_597c3f628cb0ec89` | fresh_draw_20260606 | informational | If a recreational drug is legalised do those in prison convicted of charges relating to that drug still have to serve th… | informational, non-technical, law and government, question, fact, research, sens |
| 5 | `gb_eae605f6a822e2dd` | fresh_draw_20260606 | informational | evolution of fashion trends throughout history | informational, complex, arts and entertainment, historical, non-technical, state |
| 6 | `gb_545ccaa98efb3990` | fresh_draw_20260606 | informational | Should the voting age be altered? | intermediate, debate, politics, non-technical, question, opinion, research, sens |
| 7 | `gb_4a80799ff1b73123` | fresh_draw_20260606 | informational | What is the use of magic? | simple, informational, non-technical, question, fact, research, non-sensitive, a |
| 8 | `gb_4ab106d51f3c287e` | fresh_draw_20260606 | informational | Invent a new idiom---then spill the beans. | simple, informational, books and literature, non-technical, command, non-sensiti |
| 9 | `gb_6768d049702fbc51` | fresh_draw_20260606 | informational | how to bounce back from a layoff | informational, intermediate, jobs and education, evergreen, research, non-sensit |
| 10 | `gb_d35d146fa97c16bb` | fresh_draw_20260606 | informational | who sang a whiter shade of pale first | informational, arts and entertainment, non-technical, question, fact, simple, hi |

---
*This file is for human visual inspection only. The authoritative record is the parquet.*