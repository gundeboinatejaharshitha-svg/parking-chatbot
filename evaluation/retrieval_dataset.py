"""Retrieval evaluation dataset: parking questions with hand-labeled relevant chunks.

Ground truth was built by inspecting the real chunk IDs produced by
``app.data.ingest.load_static_chunks()`` for the 10 documents under
``app/data/static/`` (one chunk per FAQ-style question/answer pair), then
writing *paraphrased* questions - not copies of the chunk's own heading - so
retrieval is actually exercised rather than trivially matched on exact text.

Each entry's ``relevant_ids`` lists the chunk ID(s) a human would judge
relevant to the question. Most questions map to exactly one chunk, since the
source documents are themselves one-question-per-chunk; a couple of entries
deliberately have two relevant chunks (the question spans two FAQ entries) or
zero (the question is outside the knowledge base, to exercise the "no
relevant documents" case required by the evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuery:
    """One evaluation question with its ground-truth relevant chunk IDs."""

    query_id: str
    query: str
    category: str
    relevant_ids: tuple[str, ...]


RETRIEVAL_DATASET: tuple[EvalQuery, ...] = (
    # ---- parking rules --------------------------------------------------------
    EvalQuery(
        "rules-01",
        "What are the parking rules I need to follow here?",
        "parking_rules",
        ("03_parking_rules::01-what-are-the-standard-parking-rules::0",),
    ),
    EvalQuery(
        "rules-02",
        "Where am I not allowed to stop or park?",
        "parking_rules",
        ("03_parking_rules::02-where-is-stopping-or-parking-prohibited::0",),
    ),
    EvalQuery(
        "rules-03",
        "What happens if I leave my car outside a marked bay?",
        "parking_rules",
        ("03_parking_rules::03-what-happens-if-i-park-outside-a-marked::0",),
    ),
    # ---- parking charges / payment ---------------------------------------------
    EvalQuery(
        "charges-01",
        "What payment methods do you accept for parking charges?",
        "parking_charges",
        ("06_payment_methods::01-which-payment-methods-are-accepted::0",),
    ),
    EvalQuery(
        "charges-02",
        "Can I settle the parking charge online instead of at the barrier?",
        "parking_charges",
        ("06_payment_methods::02-do-i-pay-at-the-barrier-or-online::0",),
    ),
    EvalQuery(
        "charges-03",
        "What should I do if my card is declined when paying for parking?",
        "parking_charges",
        ("06_payment_methods::03-what-if-my-payment-fails::0",),
    ),
    # ---- location information --------------------------------------------------
    EvalQuery(
        "location-01",
        "Where exactly is this parking facility located?",
        "location",
        ("02_location_directions::01-where-is-citypark-central-located::0",),
    ),
    EvalQuery(
        "location-02",
        "Which entrance should I use if I'm arriving from the train station?",
        "location",
        ("02_location_directions::02-which-entrance-is-closest-to-the-station::0",),
    ),
    EvalQuery(
        "location-03",
        "How can I drive up to the higher parking levels?",
        "location",
        ("02_location_directions::04-how-do-drivers-reach-the-upper-levels-of::0",),
    ),
    # ---- booking instructions ---------------------------------------------------
    EvalQuery(
        "booking-01",
        "How do I make a parking reservation and what information is needed?",
        "booking_instructions",
        ("04_booking_process::01-how-do-i-make-a-reservation-at-citypark::0",),
    ),
    EvalQuery(
        "booking-02",
        "What do I need to show at the gate to enter the car park?",
        "booking_instructions",
        ("04_booking_process::02-what-do-i-need-to-check-in-and-enter::0",),
    ),
    EvalQuery(
        "booking-03",
        "What is a booking reference used for and where do I find it?",
        "booking_instructions",
        ("04_booking_process::03-what-is-my-booking-reference-and-how-is::0",),
    ),
    # ---- cancellation policies ---------------------------------------------------
    EvalQuery(
        "cancel-01",
        "If I cancel my reservation, will I get my money back?",
        "cancellation_policy",
        (
            "05_cancellation_policy::01-how-late-can-i-cancel-my-booking-for-fre::0",
            "05_cancellation_policy::02-can-i-get-a-refund::0",
        ),
    ),
    EvalQuery(
        "cancel-02",
        "What happens if I simply don't show up for my booking?",
        "cancellation_policy",
        ("05_cancellation_policy::03-what-happens-if-i-do-not-show-up-for-a-b::0",),
    ),
    EvalQuery(
        "cancel-03",
        "Can I change the date or time on an existing booking?",
        "cancellation_policy",
        ("05_cancellation_policy::04-can-i-change-my-booking::0",),
    ),
    # ---- general parking information --------------------------------------------
    EvalQuery(
        "general-01",
        "What kind of facility is CityPark Central?",
        "general_information",
        ("01_general_info::01-what-is::0",),
    ),
    EvalQuery(
        "general-02",
        "Is there a maximum vehicle height for entering the car park?",
        "general_information",
        ("01_general_info::02-what-is-the-maximum-vehicle-height::0",),
    ),
    EvalQuery(
        "general-03",
        "Do you have charging points for electric vehicles?",
        "general_information",
        ("08_ev_charging::01-where-are-ev-bays-at-citypark-central-an::0",),
    ),
    EvalQuery(
        "general-04",
        "Is the car park monitored by security cameras?",
        "general_information",
        ("07_security_safety::01-are-there-cameras-and-patrols::0",),
    ),
    EvalQuery(
        "general-05",
        "Where can I find the accessible parking bays?",
        "general_information",
        ("09_accessibility_and_services::01-where-are-the-disabled-bays::0",),
    ),
    EvalQuery(
        "general-06",
        "What should I do if I've lost my booking reference?",
        "general_information",
        ("10_faq::01-what-should-i-do-if-i-lose-my-booking-re::0",),
    ),
    EvalQuery(
        "general-07",
        "Are pets allowed inside the car park?",
        "general_information",
        ("10_faq::03-are-pets-allowed::0",),
    ),
    # ---- no relevant document (out of knowledge-base scope) ---------------------
    EvalQuery("no-match-01", "What is the capital of France?", "out_of_scope", ()),
    EvalQuery(
        "no-match-02",
        "Can you recommend a good restaurant in the city center?",
        "out_of_scope",
        (),
    ),
)


__all__ = ["RETRIEVAL_DATASET", "EvalQuery"]
