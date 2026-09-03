"""
Lightweight local RAG knowledge base for Alzheimer's staging context.

No vector database needed for a corpus this small - uses TF-IDF + cosine
similarity (scikit-learn), which is fast, dependency-light, and fully local.

Content is written for a CLINICAL audience (referring physician / neurologist),
not patients - concise staging criteria, CDR/MMSE correlates, and imaging
findings rather than lay explanations or lifestyle advice. Retrieved to
ground the LLM's report, not to replace clinical judgment.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Knowledge base - chunked by stage. Each stage has: clinical definition,
# exam/imaging findings, and management notes.
# ---------------------------------------------------------------------------
DOCUMENTS = [
    # ---------------- NonDemented ----------------
    {
        "id": "nondemented_definition",
        "class": "NonDemented",
        "text": ("NonDemented corresponds to CDR 0. No objective cognitive "
                 "impairment; MMSE typically 26-30. Activities of daily living "
                 "(ADLs) and instrumental ADLs fully intact. Any subjective "
                 "memory complaints are not corroborated by cognitive testing."),
    },
    {
        "id": "nondemented_findings",
        "class": "NonDemented",
        "text": ("Structural MRI shows age-appropriate volumes with no "
                 "significant medial temporal lobe (hippocampal/entorhinal) "
                 "atrophy beyond expected norms for age. No abnormal white "
                 "matter changes attributable to a neurodegenerative process."),
    },
    {
        "id": "nondemented_management",
        "class": "NonDemented",
        "text": ("No intervention indicated beyond routine health "
                 "maintenance. Reassess if new cognitive complaints arise; "
                 "consider baseline cognitive screening if strong family "
                 "history or vascular risk factors are present."),
    },

    # ---------------- VeryMildDemented ----------------
    {
        "id": "verymild_definition",
        "class": "VeryMildDemented",
        "text": ("VeryMildDemented corresponds to CDR 0.5 / amnestic mild "
                 "cognitive impairment (MCI). MMSE typically 24-29. Mild, "
                 "consistent episodic memory deficit reported by patient and/or "
                 "informant, often not evident on brief exam. IADLs largely "
                 "preserved with minimal or no functional impact."),
    },
    {
        "id": "verymild_findings",
        "class": "VeryMildDemented",
        "text": ("Neuropsychological testing may show mild impairment in "
                 "delayed recall with otherwise intact domains. MRI may show "
                 "early, subtle hippocampal or entorhinal cortex volume loss; "
                 "findings can be within normal limits at this stage."),
    },
    {
        "id": "verymild_management",
        "class": "VeryMildDemented",
        "text": ("Recommend structured follow-up (cognitive reassessment "
                 "every 6-12 months), evaluation of reversible contributors "
                 "(B12, TSH, depression, sleep apnea, polypharmacy), and "
                 "vascular risk factor optimization. Consider biomarker "
                 "work-up if progression is a clinical concern."),
    },

    # ---------------- MildDemented ----------------
    {
        "id": "mild_definition",
        "class": "MildDemented",
        "text": ("MildDemented corresponds to CDR 1 (mild/early-stage "
                 "dementia). MMSE typically ~20-24. Cognitive deficits are "
                 "apparent to family and clinician and begin to interfere with "
                 "complex IADLs (finances, medication management, driving)."),
    },
    {
        "id": "mild_findings",
        "class": "MildDemented",
        "text": ("Clinical picture includes word-finding difficulty, impaired "
                 "new learning, mild executive dysfunction, and difficulty "
                 "with multi-step tasks. MRI commonly shows medial temporal "
                 "lobe atrophy (hippocampus, entorhinal cortex); may correlate "
                 "with amyloid/tau pathology if biomarker-confirmed."),
    },
    {
        "id": "mild_management",
        "class": "MildDemented",
        "text": ("Consider initiating a cholinesterase inhibitor per current "
                 "guidelines. Begin care planning: safety assessment (driving, "
                 "medication management), advance directives, and caregiver "
                 "education. Reassess functional status every 6 months."),
    },

    # ---------------- ModerateDemented ----------------
    {
        "id": "moderate_definition",
        "class": "ModerateDemented",
        "text": ("ModerateDemented corresponds to CDR 2 (moderate/middle-"
                 "stage dementia). MMSE typically ~10-19. Significant "
                 "functional decline; patient requires assistance with basic "
                 "ADLs (dressing, hygiene, toileting) in addition to IADLs."),
    },
    {
        "id": "moderate_findings",
        "class": "ModerateDemented",
        "text": ("Clinical picture includes disorientation to time/place, "
                 "impaired judgment, behavioral and psychological symptoms "
                 "(agitation, wandering, sleep disturbance), and often urinary "
                 "incontinence. MRI typically shows more pronounced medial "
                 "temporal and cortical atrophy with ventricular enlargement."),
    },
    {
        "id": "moderate_management",
        "class": "ModerateDemented",
        "text": ("Optimize symptomatic pharmacotherapy (cholinesterase "
                 "inhibitor +/- memantine per guidelines); address behavioral "
                 "symptoms with non-pharmacologic strategies first. Prioritize "
                 "home safety (fall/wandering risk), caregiver support "
                 "resources, and evaluation for formal care services."),
    },

    # ---------------- General / cross-stage ----------------
    {
        "id": "general_disclaimer",
        "class": "general",
        "text": ("Imaging-based classification is a screening/research aid "
                 "and does not constitute a clinical diagnosis. Formal "
                 "diagnosis requires correlation with standardized cognitive "
                 "testing (e.g., MMSE, MoCA), CDR staging, functional history, "
                 "and, where indicated, CSF or PET biomarkers, per current "
                 "diagnostic guidelines (e.g., NIA-AA criteria)."),
    },
]


class SimpleRAG:
    def __init__(self, documents=DOCUMENTS):
        self.documents = documents
        self.texts = [d["text"] for d in documents]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, predicted_class, top_k=4):
        """Retrieve top_k most relevant snippets, biased toward the
        predicted class + always-relevant 'general' docs."""
        candidates = [d for d in self.documents if d["class"] in (predicted_class, "general")]
        if not candidates:
            candidates = self.documents

        candidate_texts = [d["text"] for d in candidates]
        query_vec = self.vectorizer.transform([query])
        candidate_vecs = self.vectorizer.transform(candidate_texts)
        sims = cosine_similarity(query_vec, candidate_vecs)[0]

        ranked = sorted(zip(candidates, sims), key=lambda x: -x[1])
        top = [c for c, s in ranked[:top_k]]
        return top
