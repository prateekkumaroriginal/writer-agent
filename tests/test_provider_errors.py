"""Tests for retryable provider-failure classification."""

import unittest

from writer_agent.provider_errors import (
    RetryableProviderError,
    raise_if_retryable_provider_error,
)


class ProviderErrorTests(unittest.TestCase):
    def test_rate_limit_is_retryable_without_exposing_provider_payload(self):
        error = RuntimeError(
            "429 rate limit reached for organization secret-org-id"
        )
        error.status_code = 429

        with self.assertRaisesRegex(
            RetryableProviderError,
            "temporarily unavailable",
        ) as raised:
            raise_if_retryable_provider_error(error)

        self.assertNotIn("secret-org-id", str(raised.exception))

    def test_validation_error_remains_non_retryable(self):
        raise_if_retryable_provider_error(
            ValueError("Structured response was invalid.")
        )


if __name__ == "__main__":
    unittest.main()
