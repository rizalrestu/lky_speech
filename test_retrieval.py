"""Recall@K and MRR for core.retrieve() over a hand-labeled test set."""
from core import TOP_K, load_resources, retrieve

# (question, expected uid). Only records with a PDF attachment reach the corpus,
# so .htm-only records can never be retrieved — don't add them here.
TEST_SET = [
    # Political Study Centre seminar, 16 Aug 1964
    ("Why do multi-racial societies formed under colonial rule tend to splinter apart "
     "once they become independent?",
     "72f820c8-115d-11e3-83d5-0050568939ad"),

    # Joint Campus talk on bilingualism, 5 Jan 1979
    ("What are the practical limits of bilingualism, and how wide is the gap between "
     "pupils taking a language as their first versus second language?",
     "730c8847-115d-11e3-83d5-0050568939ad"),

    # Joint meeting ASSOCHAM/FICCI/CII, New Delhi, 5 Jan 1996
    ("What is the danger facing India's economic reforms, and what growth rate could "
     "India sustain if it stays the course?",
     "72f7ab37-115d-11e3-83d5-0050568939ad"),

    # Third Summit Conference of Non-Aligned Countries, Lusaka, 9 Sep 1970
    ("How did the beginning of direct dialogue between the superpowers change what "
     "non-alignment could win for newly independent countries?",
     "727bc190-115d-11e3-83d5-0050568939ad"),

    # Motion on the Malaysia Agreement, Legislative Assembly, 30 Jul 1963
    ("What did Singapore gain and what did it concede in the final Malaysia Agreement "
     "reached in London?",
     "73e69f31-115d-11e3-83d5-0050568939ad"),

    # World Economic Forum Summit keynote, 20 Sep 1995
    ("Should China be integrated into the global system, or should its access to "
     "markets, capital and technology be restricted?",
     "72f062e4-115d-11e3-83d5-0050568939ad"),

    # Everton Park Housing Estate opening, 8 Nov 1965
    ("Why should workers be able to buy their flats by instalment instead of only "
     "renting them?",
     "7433ed34-115d-11e3-83d5-0050568939ad"),

    # NYT/IHT interview with Seth Mydans, 1 Sep 2010
    ("How do you cope with growing old, and what led you to take up meditation?",
     "80e4563e-115d-11e3-83d5-0050568939ad"),

    # Narayana Murthy book launch "A Better India: A Better World", 11 May 2009
    ("What makes Narayana Murthy and Infosys exceptional compared with most Indian "
     "entrepreneurs?",
     "8023f350-115d-11e3-83d5-0050568939ad"),

    # SMU Tate Lecture Series, Dallas, 19 Oct 2006
    ("What role has the United States played in East Asia's economic revival?",
     "7ea6687b-115d-11e3-83d5-0050568939ad"),
]


def main():
    if not TEST_SET:
        print("TEST_SET is empty — add some (question, uid) pairs in this file first.")
        return

    embedder, client = load_resources()

    hits, mrr_total = 0, 0.0
    for question, expected_uid in TEST_SET:
        uids = [h.payload.get("uid") for h in retrieve(question, embedder, client)]

        rank = uids.index(expected_uid) + 1 if expected_uid in uids else None
        if rank:
            hits += 1
            mrr_total += 1 / rank
        status = f"FOUND at rank {rank}" if rank else f"NOT FOUND in top-{TOP_K}"
        print(f"- {question!r}\n  expected uid={expected_uid} -> {status}")

    n = len(TEST_SET)
    print(f"\nRecall@{TOP_K}: {hits}/{n} ({hits/n:.0%})")
    print(f"MRR: {mrr_total/n:.3f}")


if __name__ == "__main__":
    main()
