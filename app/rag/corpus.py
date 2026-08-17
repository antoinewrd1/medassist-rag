"""The retrieval corpus.

PROVENANCE, STATED PLAINLY: these passages were written for this project. They
paraphrase the *shape* of public health and clinical guidance -- the kind of
advice CDC, WHO, or a health system triage protocol publishes -- but they are
not quotations from any of those bodies and carry no clinical authority.

That distinction matters more here than in most RAG demos. A retrieval system
whose citations look official but are invented is worse than one with no
citations at all, because the citation is what earns the reader's trust. So
every document carries an explicit `provenance` field, the API returns it with
every citation, and the UI renders it next to the text.

Replacing this corpus with real ingested guidance (CDC pages, an organization's
own triage protocols) is a data problem, not a code problem -- `index.py` reads
whatever `DOCUMENTS` contains.
"""

from __future__ import annotations

PROVENANCE = "Synthetic — written for this project, not authoritative guidance"

DOCUMENTS: list[dict] = [
    {
        "id": "doc-uri-selfcare",
        "title": "Upper respiratory infection: home care and when to escalate",
        "topic": "respiratory",
        "text": (
            "Most upper respiratory infections are viral and resolve within seven to "
            "ten days without antibiotics. Supportive care includes rest, fluids, and "
            "over-the-counter symptom relief appropriate to age. Escalate to a "
            "clinician if fever persists beyond four days, if symptoms improve and "
            "then worsen, if breathing becomes difficult, or if the person has a "
            "chronic lung condition or is immunocompromised."
        ),
    },
    {
        "id": "doc-influenza",
        "title": "Influenza: antiviral window and high-risk groups",
        "topic": "respiratory",
        "text": (
            "Influenza typically presents with abrupt onset of fever, cough, muscle "
            "aches, and fatigue. Antiviral treatment is most effective when started "
            "within 48 hours of symptom onset, so people at higher risk should seek "
            "care early rather than waiting. Higher-risk groups include adults over "
            "65, pregnant people, young children, and those with chronic heart, lung, "
            "or immune conditions."
        ),
    },
    {
        "id": "doc-gastro",
        "title": "Gastroenteritis: hydration and dehydration warning signs",
        "topic": "gastrointestinal",
        "text": (
            "Acute gastroenteritis usually causes diarrhea and vomiting lasting one to "
            "three days. The main risk is dehydration, not the infection itself. Oral "
            "rehydration with fluids containing both salt and sugar is preferred over "
            "water alone. Warning signs requiring same-day care include inability to "
            "keep fluids down, markedly reduced urination, dizziness on standing, "
            "blood in the stool, or symptoms lasting more than three days."
        ),
    },
    {
        "id": "doc-foodborne",
        "title": "Suspected foodborne illness: reporting and investigation",
        "topic": "gastrointestinal",
        "text": (
            "Suspected foodborne illness should be reported to the local health "
            "department, particularly when several people who shared a meal become ill. "
            "A food history covering the 72 hours before symptom onset helps identify "
            "a common source. Stool culture may be requested to identify the organism "
            "and to link cases during an outbreak investigation."
        ),
    },
    {
        "id": "doc-headache",
        "title": "Headache: features that distinguish benign from serious",
        "topic": "neurological",
        "text": (
            "Most headaches are tension-type or migrainous and can be managed at home. "
            "Features that warrant urgent evaluation include a headache that reaches "
            "maximum intensity within seconds, a headache clearly different from a "
            "person's usual pattern, headache with fever and neck stiffness, headache "
            "following head injury, or headache accompanied by weakness, vision "
            "changes, or confusion."
        ),
    },
    {
        "id": "doc-chest-pain",
        "title": "Chest pain: why it is treated as cardiac until proven otherwise",
        "topic": "cardiac",
        "text": (
            "Chest pain has many benign causes, including musculoskeletal strain and "
            "reflux, but it cannot be distinguished from cardiac causes by description "
            "alone. Pressure or tightness, radiation to the arm, jaw, or back, and "
            "associated sweating, nausea, or breathlessness raise concern. Emergency "
            "evaluation is the default because the diagnostic test that rules out a "
            "cardiac cause is not available outside a clinical setting."
        ),
    },
    {
        "id": "doc-rash",
        "title": "Rash: distinguishing patterns that need same-day review",
        "topic": "dermatologic",
        "text": (
            "Most rashes are self-limited and can be reviewed routinely. A rash that "
            "does not fade under gentle pressure, a rash with fever and severe illness, "
            "blistering involving the mouth or eyes, or rapidly spreading redness with "
            "pain and warmth all warrant same-day assessment. Photographing the rash on "
            "first appearance helps a clinician judge how quickly it is progressing."
        ),
    },
    {
        "id": "doc-uti",
        "title": "Urinary symptoms: routine versus escalating presentations",
        "topic": "genitourinary",
        "text": (
            "Burning on urination, urinary frequency, and urgency commonly indicate a "
            "lower urinary tract infection and are usually managed with a clinician "
            "assessment and, where appropriate, antibiotics. Flank pain, fever, chills, "
            "or vomiting suggest upper tract involvement and need prompt evaluation. "
            "Urinary symptoms in pregnancy should always be assessed rather than "
            "monitored at home."
        ),
    },
    {
        "id": "doc-pediatric-fever",
        "title": "Fever in children: age-dependent thresholds",
        "topic": "pediatric",
        "text": (
            "Fever thresholds in children depend heavily on age. Any fever in an infant "
            "under three months requires immediate evaluation regardless of how well "
            "the infant appears. In older children, the child's behavior matters more "
            "than the number on the thermometer: poor fluid intake, unusual drowsiness, "
            "difficulty rousing, or a decline in responsiveness are more concerning "
            "than the temperature alone."
        ),
    },
    {
        "id": "doc-mental-health",
        "title": "Emotional distress: immediate support options",
        "topic": "behavioral",
        "text": (
            "Emotional distress, hopelessness, and thoughts of self-harm are medical "
            "concerns and are treatable. In the United States, the 988 Suicide and "
            "Crisis Lifeline provides free confidential support by call or text, 24 "
            "hours a day. Immediate danger warrants emergency services. Ongoing support "
            "from a primary care clinician or mental health professional is appropriate "
            "even when the situation is not an emergency."
        ),
    },
    {
        "id": "doc-when-ed",
        "title": "Choosing between emergency, urgent, and routine care",
        "topic": "general",
        "text": (
            "Emergency departments are appropriate for symptoms that could threaten "
            "life, limb, or sight within hours, including chest pain, stroke signs, "
            "breathing difficulty, and uncontrolled bleeding. Urgent care suits "
            "problems needing attention the same day but not immediately, such as "
            "possible fractures, moderate infections, or persistent vomiting. Routine "
            "primary care is appropriate for stable symptoms present for days to weeks."
        ),
    },
    {
        "id": "doc-antibiotics",
        "title": "Antibiotic stewardship in common infections",
        "topic": "general",
        "text": (
            "Antibiotics treat bacterial infections and have no effect on viral "
            "illnesses such as the common cold, most sore throats, and influenza. "
            "Unnecessary use drives resistance and causes avoidable side effects. "
            "Clinicians weigh symptom duration, severity, and test results before "
            "prescribing, which is why a request for antibiotics without an assessment "
            "is usually declined."
        ),
    },
]

for _doc in DOCUMENTS:
    _doc.setdefault("provenance", PROVENANCE)
