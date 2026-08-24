"""Offline regression tests for the Day 22 lab implementation."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# Unit tests must never emit LangSmith traces or call a hosted service.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["DAY22_ENV_OVERRIDE"] = "false"


def load_step(module_name: str, filename: str):
    """Load a step module whose filename starts with a number."""
    spec = importlib.util.spec_from_file_location(module_name, SRC / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


step1 = load_step("lab_step1", "01_langsmith_rag_pipeline.py")
step2 = load_step("lab_step2", "02_prompt_hub_ab_routing.py")
step3 = load_step("lab_step3", "03_ragas_evaluation.py")
step4 = load_step("lab_step4", "04_guardrails_validator.py")
run_all = load_step("lab_run_all", "run_all.py")

# config.py intentionally prioritizes the lab's .env. Re-disable tracing after
# importing it so chain invocations below remain strictly offline.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from qa_pairs import QA_PAIRS, SAMPLE_QUESTIONS


class _FakeVectorStore:
    def as_retriever(self, search_kwargs):
        if search_kwargs != {"k": 3}:
            raise AssertionError("The lab retriever must request top-3 chunks")
        return RunnableLambda(
            lambda _question: [
                Document(page_content="Grounded fact one."),
                Document(page_content="Grounded fact two."),
            ]
        )


class LabTests(unittest.TestCase):
    def test_exactly_50_aligned_qa_pairs(self):
        self.assertEqual(len(SAMPLE_QUESTIONS), 50)
        self.assertEqual(len(QA_PAIRS), 50)
        self.assertEqual(
            SAMPLE_QUESTIONS,
            [sample["question"] for sample in QA_PAIRS],
        )

    def test_step1_chain_returns_traceable_triplet(self):
        fake_llm = FakeListChatModel(responses=["A grounded answer."])
        with patch.object(step1, "get_llm", return_value=fake_llm):
            chain, _retriever = step1.build_rag_chain(_FakeVectorStore())

        output = chain.invoke("What is the fact?")

        self.assertEqual(output["question"], "What is the fact?")
        self.assertIn("Grounded fact one.", output["context"])
        self.assertEqual(output["answer"], "A grounded answer.")

    def test_ab_routing_is_deterministic_and_uses_both_versions(self):
        request_ids = [f"req-{index:04d}" for index in range(50)]
        first_pass = [step2.get_prompt_version(value) for value in request_ids]
        second_pass = [step2.get_prompt_version(value) for value in request_ids]

        self.assertEqual(first_pass, second_pass)
        self.assertIn(step2.PROMPT_V1_NAME, first_pass)
        self.assertIn(step2.PROMPT_V2_NAME, first_pass)

    def test_prompt_hub_names_are_owner_qualified(self):
        class FakeSettings:
            tenant_handle = "student-handle"

        class FakeClient:
            @staticmethod
            def _get_settings():
                return FakeSettings()

        with patch.object(step2.config, "LANGSMITH_PROMPT_OWNER", ""):
            names = step2.resolve_hub_prompt_names(FakeClient())

        self.assertEqual(
            names["v1"], f"student-handle/{step2.PROMPT_V1_NAME}"
        )
        self.assertEqual(
            names["v2"], f"student-handle/{step2.PROMPT_V2_NAME}"
        )

    def test_private_prompt_names_work_without_public_handle(self):
        class FakeSettings:
            tenant_handle = None

        class FakeClient:
            @staticmethod
            def _get_settings():
                return FakeSettings()

        with patch.object(step2.config, "LANGSMITH_PROMPT_OWNER", ""):
            names = step2.resolve_hub_prompt_names(FakeClient())

        self.assertEqual(names["v1"], step2.PROMPT_V1_NAME)
        self.assertEqual(names["v2"], step2.PROMPT_V2_NAME)

    def test_ragas_dataset_uses_required_single_turn_fields(self):
        dataset = step3.build_ragas_dataset(
            [
                {
                    "question": "Question",
                    "answer": "Answer",
                    "contexts": ["Context one", "Context two"],
                    "reference": "Reference",
                }
            ]
        )

        sample = dataset.samples[0]
        self.assertEqual(sample.user_input, "Question")
        self.assertEqual(sample.response, "Answer")
        self.assertEqual(sample.retrieved_contexts, ["Context one", "Context two"])
        self.assertEqual(sample.reference, "Reference")

    def test_pii_detector_redacts_all_supported_types(self):
        value = (
            "Email user@example.com, phone 555-123-4567, "
            "SSN 123-45-6789, card 4111 1111 1111 1111."
        )
        result = step4.PIIDetector().validate(value, {})

        self.assertEqual(
            result.fix_value,
            "Email [EMAIL_REDACTED], phone [PHONE_REDACTED], "
            "SSN [SSN_REDACTED], card [CREDIT_CARD_REDACTED].",
        )

    def test_json_formatter_repairs_and_falls_back_to_valid_json(self):
        formatter = step4.JSONFormatter()
        repaired = formatter.validate("```json\n{'name': 'Ada',}\n```", {})
        fallback = formatter.validate("not json {]", {})

        self.assertEqual(json.loads(repaired.fix_value), {"name": "Ada"})
        self.assertEqual(json.loads(fallback.fix_value)["error"], "Không thể phân tích JSON")

    def test_run_all_returns_nonzero_when_a_step_fails(self):
        with patch.object(run_all, "run_step", return_value=False), patch.object(
            sys, "argv", ["run_all.py", "--step", "1"]
        ):
            with redirect_stdout(StringIO()):
                self.assertEqual(run_all.main(), 1)


if __name__ == "__main__":
    unittest.main()
