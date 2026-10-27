import json

from backend.app.prompts.citation_verification_prompt import (
    SYSTEM_PROMPT,
    build_verification_prompt,
)
from backend.app.services.llm_service import LLMService


class CitationVerificationService:
    """
    Verifies whether generated answer claims are supported
    by retrieved research-paper evidence.
    """

    def __init__(
        self,
        llm_service: LLMService | None = None,
    ) -> None:

        self.llm_service = (
            llm_service
            if llm_service is not None
            else LLMService()
        )

    def verify(
        self,
        answer: str,
        evidence: list[dict],
    ) -> dict:
        """
        Verify a generated answer against retrieved evidence.
        """

        if not answer.strip():
            raise ValueError(
                "Answer cannot be empty."
            )

        if not evidence:
            raise ValueError(
                "Evidence cannot be empty."
            )

        verification_prompt = (
            build_verification_prompt(
                answer=answer,
                evidence=evidence,
            )
        )

        raw_response = (
            self.llm_service.generate_answer(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=verification_prompt,
                max_tokens=1500,
            )
        )

        try:

            verification_result = json.loads(
                raw_response
            )

        except json.JSONDecodeError as error:

            print("\n" + "=" * 60)
            print(
                "CITATION VERIFICATION RAW RESPONSE"
            )
            print("=" * 60)
            print(repr(raw_response))
            print("=" * 60)

            raise ValueError(
                "Citation verification returned "
                "invalid JSON."
            ) from error

        self._validate_result(
            verification_result
        )

        return verification_result

    def _validate_result(
        self,
        result: dict,
    ) -> None:
        """
        Validate the structure returned by the verifier.
        """

        if not isinstance(
            result,
            dict,
        ):
            raise ValueError(
                "Citation verification result "
                "must be a JSON object."
            )

        if "claims" not in result:
            raise ValueError(
                "Citation verification result "
                "is missing 'claims'."
            )

        if "overall_status" not in result:
            raise ValueError(
                "Citation verification result "
                "is missing 'overall_status'."
            )

        if not isinstance(
            result["claims"],
            list,
        ):
            raise ValueError(
                "'claims' must be a list."
            )

        allowed_statuses = {
            "SUPPORTED",
            "PARTIALLY_SUPPORTED",
            "UNSUPPORTED",
        }

        if result["overall_status"] not in (
            allowed_statuses
        ):
            raise ValueError(
                "Invalid overall verification status."
            )

        for claim in result["claims"]:

            if not isinstance(
                claim,
                dict,
            ):
                raise ValueError(
                    "Each claim must be a JSON object."
                )

            required_fields = {
                "claim",
                "status",
                "evidence_indexes",
                "reason",
            }

            missing_fields = (
                required_fields
                - claim.keys()
            )

            if missing_fields:

                raise ValueError(
                    "Verification claim is missing "
                    f"fields: {missing_fields}"
                )

            if claim["status"] not in (
                allowed_statuses
            ):
                raise ValueError(
                    f"Invalid claim status: "
                    f"{claim['status']}"
                )

            if not isinstance(
                claim["evidence_indexes"],
                list,
            ):
                raise ValueError(
                    "'evidence_indexes' must be a list."
                )

            if not isinstance(
                claim["claim"],
                str,
            ):
                raise ValueError(
                    "'claim' must be a string."
                )

            if not isinstance(
                claim["reason"],
                str,
            ):
                raise ValueError(
                    "'reason' must be a string."
                )