"""Plain-language clinician guidance for the local CARE-AI prototype.

This module deliberately produces a review checklist, not a diagnosis, test
order, or prescription. A chest X-ray classifier cannot safely choose treatment
without examination, history, medication/allergy review, and local clinical
protocols.
"""

from __future__ import annotations

from typing import Any


REFERENCE_LINKS = [
    {
        "title": "WHO: Ethics and governance of AI for health",
        "url": "https://www.who.int/publications/i/item/9789240029200",
        "note": "Human oversight, transparency, safety, and accountability.",
    },
    {
        "title": "CDC: Clinical and laboratory diagnosis of tuberculosis",
        "url": "https://www.cdc.gov/tb/hcp/testing-diagnosis/clinical-and-laboratory-diagnosis.html",
        "note": "Chest radiography is one part of a broader evaluation when TB is a concern.",
    },
    {
        "title": "ACR Appropriateness Criteria: Acute respiratory illness",
        "url": "https://acsearch.acr.org/docs/69446/narrative/",
        "note": "Imaging choices depend on the clinical scenario; local guidance takes precedence.",
    },
]


PLAIN_LABELS = {
    "Atelectasis": "Possible partial collapse of lung tissue",
    "Cardiomegaly": "Possible enlarged heart silhouette",
    "Consolidation": "Possible dense air-space change",
    "Edema": "Possible fluid-related lung change",
    "Enlarged Cardiomediastinum": "Possible widening near the heart and central chest",
    "Fracture": "Possible fracture-related finding",
    "Lung Lesion": "Possible focal lung lesion",
    "Lung Opacity": "Possible area of increased whiteness in the lung",
    "No Finding": "No strong abnormal finding signal",
    "Pleural Effusion": "Possible fluid around the lung",
    "Pleural Other": "Possible pleural-space finding",
    "Pneumonia": "Possible infection-like lung pattern",
    "Pneumothorax": "Possible air around the lung",
    "Support Devices": "Possible tube or line requiring position review",
}


def _present(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "no", "none", "null"}


def _any_present(clinical: dict[str, Any], *keys: str) -> bool:
    return any(_present(clinical.get(key)) for key in keys)


def _number(clinical: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        try:
            return float(clinical[key])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def build_clinical_guidance(
    *,
    labels: list[dict[str, Any]],
    clinical: dict[str, Any],
    quality_score: float,
    clinical_score: float,
    support_score: float,
    predictive_uncertainty: float,
    review_priority: str,
    demo_mode: bool,
) -> dict[str, Any]:
    """Build safe, clinician-facing next-step guidance from model output.

    The returned ``possible_tests`` are discussion prompts only. They are not
    orders, and the treatment block intentionally never contains a medicine or
    dose.
    """

    positive = [
        item for item in labels
        if str(item.get("decision", "")).startswith("positive")
    ]
    abstained = [item for item in labels if item.get("abstained")]
    abnormal_positive = [item for item in positive if str(item.get("label", "")).lower() != "no finding"]
    no_finding_positive = [item for item in positive if str(item.get("label", "")).lower() == "no finding"]
    top_positive = sorted(abnormal_positive, key=lambda item: float(item.get("probability", 0.0)), reverse=True)
    top_label = top_positive[0] if top_positive else None
    top_name = PLAIN_LABELS.get(str(top_label.get("label")), str(top_label.get("label"))) if top_label else None

    if demo_mode:
        headline = "Demonstration mode — clinician review required"
        plain_summary = (
            "The application is running without a validated local checkpoint. "
            "It cannot provide a clinically meaningful image finding."
        )
    elif top_name:
        headline = f"Review for: {top_name}"
        plain_summary = (
            f"The AI detected a pattern that may correspond to {top_name.lower()}. "
            "This is an assistance signal, not a diagnosis. Confirm it against the original image, examination, and report."
        )
    else:
        headline = "No strong abnormal pattern detected by the AI"
        plain_summary = (
            "The AI did not detect a strong abnormal pattern in this image. "
            "A reassuring AI result does not rule out disease when symptoms, examination, prior imaging, or clinical risk remain concerning."
        )

    urgency_flags: list[str] = []
    oxygen = _number(clinical, "oxygen_saturation", "spo2", "o2_saturation")
    if _any_present(clinical, "severe_symptoms", "emergency_symptoms", "collapse", "confusion", "blue_lips"):
        urgency_flags.append("reported severe or emergency symptoms")
    if _any_present(clinical, "coughing_blood", "hemoptysis"):
        urgency_flags.append("coughing blood")
    if _any_present(clinical, "chest_pain", "acute_chest_pain"):
        urgency_flags.append("chest pain")
    if oxygen is not None and oxygen < 92:
        urgency_flags.append(f"oxygen saturation recorded as {oxygen:g}%")

    if urgency_flags:
        urgency = {
            "level": "urgent",
            "title": "Urgent clinical assessment may be needed",
            "message": (
                "Do not wait for the AI result to manage an unstable patient. "
                "Follow the facility's emergency pathway and assess the patient in person."
            ),
            "reasons": urgency_flags,
        }
    elif review_priority == "high" or predictive_uncertainty >= 0.20:
        urgency = {
            "level": "prompt_review",
            "title": "Prompt clinician review recommended",
            "message": "The result is less stable or less supported than the routine pathway. Review the image and clinical context before acting.",
            "reasons": ["low support or higher prediction uncertainty"],
        }
    else:
        urgency = {
            "level": "routine_review",
            "title": "Routine clinician review required",
            "message": "Use the AI as a second reader and document the independent clinical impression.",
            "reasons": [],
        }

    checklist = [
        "Assess the patient's current stability, vital signs, oxygenation, and examination first.",
        "Review the original radiograph for projection, rotation, exposure, and anatomic coverage; do not rely on the heatmap alone.",
        "Compare with prior imaging and the formal radiology report when available.",
        "If the AI signal conflicts with symptoms, examination, prior imaging, or image quality, send the case for qualified radiologist review.",
    ]
    if quality_score < 0.45:
        checklist.insert(1, "Image quality is limited by the proxy score; consider whether repeat acquisition is clinically appropriate before interpreting subtle findings.")
    if clinical_score < 0.45:
        checklist.append("Clinical context is incomplete; record the indication, symptom duration, relevant history, and key vitals before finalizing the assessment.")
    if abstained:
        checklist.append("One or more finding categories were withheld because support was insufficient; treat this as uncertainty, not as a negative result.")

    possible_tests: list[dict[str, str]] = []
    if _any_present(clinical, "respiratory_symptoms", "dyspnea", "shortness_of_breath", "cough", "fever", "symptoms"):
        possible_tests.append({
            "title": "Respiratory assessment",
            "text": "Confirm vital signs and oxygen saturation, then choose any laboratory or imaging follow-up from the examination and local respiratory protocol.",
        })
    if _any_present(clinical, "tb_concern", "tb_exposure", "positive_tb_test", "tuberculosis"):
        possible_tests.append({
            "title": "If tuberculosis is clinically suspected",
            "text": "Use a complete TB evaluation pathway; this may include chest-radiograph review and appropriate sputum-based microbiologic testing. Follow local or national TB guidance.",
        })
    if _any_present(clinical, "chest_pain", "acute_chest_pain"):
        possible_tests.append({
            "title": "Chest-pain pathway",
            "text": "Evaluate cardiac and other urgent causes independently. A chest X-ray AI result cannot exclude a cardiac or vascular emergency.",
        })
    if _any_present(clinical, "recent_trauma", "trauma"):
        possible_tests.append({
            "title": "If trauma is suspected",
            "text": "Consider additional radiographs or CT only when indicated by the mechanism, examination, and local imaging criteria.",
        })
    if _any_present(clinical, "immunocompromised", "immunosuppressed"):
        possible_tests.append({
            "title": "Higher-risk host",
            "text": "Use a lower threshold for specialist/radiologist review and further evaluation because a subtle or apparently reassuring radiograph may not exclude disease in every patient.",
        })
    if not possible_tests:
        possible_tests.append({
            "title": "No automatic test recommendation",
            "text": "An additional test cannot be selected safely from the image score alone. Choose investigations only after clinical assessment and a specific diagnostic question.",
        })

    treatment = {
        "title": "Medication and prescription safety",
        "status": "No prescription generated",
        "message": (
            "Do not start, stop, or dose medication from this AI output. "
            "Treatment requires a confirmed diagnosis plus review of allergies, interactions, pregnancy status, age, renal/hepatic function, comorbidities, and local guidelines."
        ),
        "clinician_action": "Document the diagnosis and treatment rationale independently of the model.",
    }

    return {
        "headline": headline,
        "plain_summary": plain_summary,
        "urgency": urgency,
        "checklist": _unique(checklist),
        "possible_tests": possible_tests,
        "treatment": treatment,
        "model_language": {
            "positive_signal": "The model noticed a pattern that deserves review; it does not mean the condition is confirmed.",
            "negative_signal": "The model did not see a strong pattern; it does not rule out disease.",
            "abstain": "The model did not have enough support to make a reliable call for this category.",
        },
        "references": REFERENCE_LINKS,
        "input_context": {
            "urgent_flags_detected": bool(urgency_flags),
            "quality_score": round(float(quality_score), 4),
            "clinical_context_score": round(float(clinical_score), 4),
            "support_score": round(float(support_score), 4),
        },
    }
