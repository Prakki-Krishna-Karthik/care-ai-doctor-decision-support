import unittest

from clinical_guidance import build_clinical_guidance


class ClinicalGuidanceTests(unittest.TestCase):
    def base(self, clinical=None, labels=None, **overrides):
        return build_clinical_guidance(
            labels=labels or [],
            clinical=clinical or {},
            quality_score=overrides.get("quality_score", 0.8),
            clinical_score=overrides.get("clinical_score", 0.7),
            support_score=overrides.get("support_score", 0.7),
            predictive_uncertainty=overrides.get("predictive_uncertainty", 0.1),
            review_priority=overrides.get("review_priority", "routine"),
            demo_mode=overrides.get("demo_mode", False),
        )

    def test_positive_signal_is_plain_language_and_never_a_prescription(self):
        guidance = self.base(labels=[{
            "label": "Pneumonia",
            "probability": 0.82,
            "decision": "positive_signal_for_review",
            "abstained": False,
        }])
        self.assertIn("infection-like", guidance["headline"])
        self.assertEqual(guidance["treatment"]["status"], "No prescription generated")
        self.assertNotIn("amoxicillin", guidance["treatment"]["message"].lower())

    def test_tb_context_adds_discussion_prompt(self):
        guidance = self.base(clinical={"tb_concern": True})
        titles = [item["title"] for item in guidance["possible_tests"]]
        self.assertIn("If tuberculosis is clinically suspected", titles)

    def test_emergency_context_raises_urgency(self):
        guidance = self.base(clinical={"chest_pain": True, "oxygen_saturation": 89})
        self.assertEqual(guidance["urgency"]["level"], "urgent")
        self.assertTrue(guidance["input_context"]["urgent_flags_detected"])

    def test_negative_signal_does_not_claim_rule_out(self):
        guidance = self.base(labels=[{
            "label": "No Finding",
            "probability": 0.2,
            "decision": "negative_signal_for_review",
            "abstained": False,
        }])
        self.assertIn("does not rule out disease", guidance["plain_summary"])

    def test_no_finding_signal_is_not_described_as_a_disease(self):
        guidance = self.base(labels=[{
            "label": "No Finding",
            "probability": 0.82,
            "decision": "positive_signal_for_review",
            "abstained": False,
        }])
        self.assertEqual(guidance["headline"], "No strong abnormal pattern detected by the AI")


if __name__ == "__main__":
    unittest.main()
