
import logging
import os
import time

from dotenv import load_dotenv
from groq import Groq

from generation.grounded import validate_citations
from generation.models import GroundedAnswer, Source

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"

ABSTENTION = (
    "I couldn't find sufficient evidence in the retrieved "
    "passages to answer this question."
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 1
RETRY_DELAYS = (1,)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class GeminiServiceError(Exception):
    """Backward-compatible service error for the generation layer."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


# Keep the existing exception name so current UI error handling
# does not need to change during the provider migration.
GenerationServiceError = GeminiServiceError


def _get_status_code(exc: Exception) -> int | None:
    """Extract an HTTP status code from API exceptions."""
    code = getattr(exc, "status_code", None)

    if code is None:
        code = getattr(exc, "code", None)

    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _generate_with_retry(client, prompt: str) -> str:
    """Generate an answer with at most one retry for transient failures."""
    for attempt in range(MAX_RETRIES + 1):
        start = time.perf_counter()

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional legal document "
                            "research assistant. Answer only from the "
                            "provided document excerpts. Do not invent "
                            "facts or citations. Cite supported claims "
                            "using [Page N]. If evidence is insufficient, "
                            f"respond exactly: {ABSTENTION}"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )

            elapsed = time.perf_counter() - start
            logger.info("Groq request succeeded in %.2fs", elapsed)

            content = response.choices[0].message.content
            return (content or "").strip()

        except Exception as exc:
            elapsed = time.perf_counter() - start
            status_code = _get_status_code(exc)

            logger.warning(
                "Groq request failed after %.2fs; attempt %s/%s; "
                "exception=%s; HTTP status=%s",
                elapsed,
                attempt + 1,
                MAX_RETRIES + 1,
                type(exc).__name__,
                status_code,
            )

            if status_code not in RETRYABLE_STATUS_CODES:
                if status_code in {401, 403}:
                    message = (
                        "Groq authentication failed. Check your API key "
                        "and account permissions."
                    )
                elif status_code == 400:
                    message = (
                        "Groq rejected the request. Check the model ID "
                        "and request parameters."
                    )
                elif status_code == 404:
                    message = (
                        "The configured Groq model was not found or is "
                        "unavailable to this account."
                    )
                else:
                    message = (
                        "Groq could not complete the request. "
                        "Check the application logs."
                    )

                raise GeminiServiceError(message) from None

            if attempt == MAX_RETRIES:
                raise GeminiServiceError(
                    "Groq is temporarily unavailable or busy. "
                    "Your document remains loaded; please try again shortly.",
                    retryable=True,
                ) from None

            time.sleep(RETRY_DELAYS[attempt])

    raise GeminiServiceError("Groq could not complete the request.")


def generate_answer(
    question: str,
    sources: list[Source],
) -> GroundedAnswer:
    if not sources:
        return GroundedAnswer(
            answer=ABSTENTION,
            sources=[],
            confidence=0.0,
            insufficient_evidence=True,
            citations_valid=False,
        )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise GeminiServiceError(
            "The Groq API key is not configured. "
            "Set GROQ_API_KEY in your .env file and restart the app."
        )

    context = "\n\n".join(
        f"[Page {source.page_number}]\n{source.text}"
        for source in sources
    )

    prompt = f"""
Answer the question using only the supplied document excerpts.

QUESTION:
{question}

DOCUMENT EXCERPTS:
{context}

RULES:
1. Use only information directly supported by the excerpts.
2. Do not infer legal obligations from unrelated passages.
3. If the excerpts do not establish the answer, output exactly:
   "{ABSTENTION}"
4. Keep supported answers concise and professional.
5. Cite factual claims using [Page N] from the supplied excerpts.
6. Never invent facts, quotations, clauses, or page numbers.
7. Do not give legal advice.
8. Do not add unrelated commentary.

ANSWER:
"""

    # Disable SDK-level retries; our function controls retry behavior.
    client = Groq(
        api_key=api_key,
        max_retries=0,
        timeout=30.0,
    )

    answer = _generate_with_retry(client, prompt)

    if not answer:
        answer = ABSTENTION

    insufficient = (
        answer.casefold().startswith(
            "i couldn't find sufficient evidence"
        )
        or answer.casefold().startswith(
            "i could not find sufficient evidence"
        )
    )

    citations_valid = (
        not insufficient and validate_citations(answer, sources)
    )

    return GroundedAnswer(
        answer=answer,
        sources=sources,
        confidence=None,
        insufficient_evidence=insufficient,
        citations_valid=citations_valid,
    )
