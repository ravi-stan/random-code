import asyncio
import aiohttp
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class CustomOcrAsyncPoller:
    """
    A custom async poller for OCR long-running operations that mimic
    Azure Document Intelligence Poller behavior.
    """

    def __init__(
        self,
        operation_url: str,
        status_url: str,
        result_url: str,
        poll_interval: float = 5.0,
        headers: Optional[Dict[str, str]] = None,
        session: aiohttp.ClientSession = None
    ):
        """
        :param str operation_url: The URL to submit the initial OCR request.
        :param str status_url: The URL to query the operation status.
        :param str result_url: The URL to retrieve the final OCR result.
        :param float poll_interval: Number of seconds to wait between status polling attempts.
        :param dict headers: Optional HTTP headers for requests.
        :param session: Optional aiohttp.ClientSession to reuse connections.
        """
        self.operation_url = operation_url
        self.status_url = status_url
        self.result_url = result_url
        self.poll_interval = poll_interval
        self.headers = headers or {}

        self._operation_id = None
        self._status = None
        self._result = None

        # If session is provided, we won't manage its lifecycle.
        # Otherwise, we'll create one on-demand in begin_operation.
        self._external_session = session
        self._session = session

        # Internal flag to check if this poller created the session
        self._own_session = False if session else True

    async def __aenter__(self):
        """
        Allows using 'async with CustomOcrAsyncPoller(...) as poller:'
        """
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._own_session = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Ensures the aiohttp session is closed if created internally.
        """
        if self._own_session and self._session:
            await self._session.close()
            self._session = None

    async def begin_operation(self, data: Dict[str, Any]) -> "CustomOcrAsyncPoller":
        """
        Starts the OCR process by calling the operation endpoint.
        Expects a response with an operation_id for further status checks.

        :param dict data: The payload to initiate the operation (e.g., file info or image reference).
        :return: self (returns the same poller instance).
        """
        if not self._session:
            # If it wasn't created and not provided, create it
            self._session = aiohttp.ClientSession()
            self._own_session = True

        logger.debug("Starting OCR operation with data: %s", data)
        async with self._session.post(
            self.operation_url,
            json=data,
            headers=self.headers
        ) as response:
            response.raise_for_status()
            content = await response.json()
            # Adjust key retrieval according to the service response
            self._operation_id = content["operation_id"]
            self._status = "inProgress"

        logger.debug("Operation started, ID: %s", self._operation_id)
        return self

    async def wait_for_completion(self) -> Dict[str, Any]:
        """
        Polls the status endpoint until the operation is complete or fails.
        Once 'succeeded', fetches the final result.

        :return: The final OCR result (JSON/dict).
        """
        if not self._operation_id:
            raise RuntimeError("Operation has not been started. Call begin_operation first.")

        while self._status not in ["succeeded", "failed"]:
            logger.debug("Current status: %s. Sleeping for %s seconds.", self._status, self.poll_interval)
            await asyncio.sleep(self.poll_interval)
            await self._update_status()

            if self._status == "succeeded":
                await self._fetch_result()
            elif self._status == "failed":
                raise RuntimeError(f"OCR operation failed (operation_id={self._operation_id}).")

        return self._result

    async def _update_status(self):
        """
        Private method to update the current status by calling the status endpoint.
        """
        url = f"{self.status_url}/{self._operation_id}"
        logger.debug("Checking status at: %s", url)
        async with self._session.get(url, headers=self.headers) as response:
            response.raise_for_status()
            status_data = await response.json()
            # Adjust key retrieval according to the service response
            self._status = status_data["status"]
            logger.debug("Status updated to: %s", self._status)

    async def _fetch_result(self):
        """
        Private method to fetch the final OCR result once the operation has succeeded.
        """
        url = f"{self.result_url}/{self._operation_id}"
        logger.debug("Fetching result from: %s", url)
        async with self._session.get(url, headers=self.headers) as response:
            response.raise_for_status()
            self._result = await response.json()
        logger.debug("Result fetched: %s", self._result)

    def status(self) -> str:
        """
        Returns the last known status of the operation.
        """
        return self._status

    def result(self) -> Dict[str, Any]:
        """
        Returns the cached result of the OCR operation (if already retrieved).
        If it is not yet retrieved, you can call wait_for_completion() to retrieve it.
        """
        if self._result is None and self._status != "succeeded":
            raise RuntimeError(
                "No result available. Call wait_for_completion() to wait for the operation to finish."
            )
        return self._result

# Example usage
# -------------
# async def main():
#     poller = CustomOcrAsyncPoller(
#         operation_url="https://your-ocr-service.com/api/ocr/start",
#         status_url="https://your-ocr-service.com/api/ocr/status",
#         result_url="https://your-ocr-service.com/api/ocr/result",
#         poll_interval=5.0,
#         headers={
#             "Authorization": "Bearer <your_token>",
#             "Content-Type": "application/json"
#         }
#     )
#
#     data = {
#         "document_url": "https://example.com/images/sample-document.png"
#     }
#
#     async with poller as p:
#         await p.begin_operation(data)
#         final_result = await p.wait_for_completion()
#         print("OCR Operation Completed. Result:", final_result)
#
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())




import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from custom_ocr_poller import CustomOcrAsyncPoller

@pytest.mark.asyncio
async def test_begin_operation():
    """Test that begin_operation sets operation_id and status."""
    mock_response = AsyncMock()
    mock_response.json.return_value = {"operation_id": "12345"}
    mock_response.raise_for_status = MagicMock()

    mock_session = AsyncMock()
    mock_session.post.return_value.__aenter__.return_value = mock_response

    # Instantiate poller with a mocked session
    poller = CustomOcrAsyncPoller(
        operation_url="http://test/operation",
        status_url="http://test/status",
        result_url="http://test/result",
        session=mock_session
    )

    await poller.begin_operation({"foo": "bar"})

    assert poller._operation_id == "12345"
    assert poller._status == "inProgress"
    mock_session.post.assert_called_once_with(
        "http://test/operation",
        json={"foo": "bar"},
        headers={}
    )

@pytest.mark.asyncio
async def test_wait_for_completion_succeeded():
    """Test wait_for_completion flow when status eventually is succeeded."""
    # Mocked statuses
    statuses = [
        {"status": "inProgress"},
        {"status": "inProgress"},
        {"status": "succeeded"}
    ]
    # Mocked final result
    final_result = {"text": "Hello World"}

    # Mock for the initial post to begin operation
    mock_begin_response = AsyncMock()
    mock_begin_response.json.return_value = {"operation_id": "12345"}
    mock_begin_response.raise_for_status = MagicMock()

    # Mock for status calls
    mock_status_response = AsyncMock()
    mock_status_response.raise_for_status = MagicMock()
    # We'll dynamically return statuses from each call
    async def status_json_side_effect():
        return statuses.pop(0)
    mock_status_response.json.side_effect = status_json_side_effect

    # Mock for the final result
    mock_result_response = AsyncMock()
    mock_result_response.raise_for_status = MagicMock()
    mock_result_response.json.return_value = final_result

    mock_session = AsyncMock()
    mock_session.post.return_value.__aenter__.return_value = mock_begin_response
    mock_session.get.return_value.__aenter__.side_effect = [
        mock_status_response,  # first status check
        mock_status_response,  # second status check
        mock_status_response,  # third status check -> "succeeded"
        mock_result_response   # final result
    ]

    poller = CustomOcrAsyncPoller(
        operation_url="http://test/operation",
        status_url="http://test/status",
        result_url="http://test/result",
        poll_interval=0.01,  # small interval for test
        session=mock_session
    )

    await poller.begin_operation({"foo": "bar"})
    result = await poller.wait_for_completion()

    assert poller._status == "succeeded"
    assert poller._result == final_result
    assert result == final_result

@pytest.mark.asyncio
async def test_wait_for_completion_failed():
    """Test wait_for_completion flow when status becomes failed."""
    statuses = [
        {"status": "inProgress"},
        {"status": "failed"}
    ]

    mock_begin_response = AsyncMock()
    mock_begin_response.json.return_value = {"operation_id": "12345"}
    mock_begin_response.raise_for_status = MagicMock()

    mock_status_response = AsyncMock()
    mock_status_response.raise_for_status = MagicMock()
    async def status_json_side_effect():
        return statuses.pop(0)
    mock_status_response.json.side_effect = status_json_side_effect

    mock_session = AsyncMock()
    mock_session.post.return_value.__aenter__.return_value = mock_begin_response
    mock_session.get.return_value.__aenter__.side_effect = [
        mock_status_response,
        mock_status_response
    ]

    poller = CustomOcrAsyncPoller(
        operation_url="http://test/operation",
        status_url="http://test/status",
        result_url="http://test/result",
        poll_interval=0.01,
        session=mock_session
    )

    await poller.begin_operation({"foo": "bar"})

    with pytest.raises(RuntimeError) as excinfo:
        await poller.wait_for_completion()

    assert "failed" in str(excinfo.value)

@pytest.mark.asyncio
async def test_result_without_waiting():
    """Test that result() raises if not succeeded yet and result not fetched."""
    mock_begin_response = AsyncMock()
    mock_begin_response.json.return_value = {"operation_id": "12345"}
    mock_begin_response.raise_for_status = MagicMock()

    mock_session = AsyncMock()
    mock_session.post.return_value.__aenter__.return_value = mock_begin_response

    poller = CustomOcrAsyncPoller(
        operation_url="http://test/operation",
        status_url="http://test/status",
        result_url="http://test/result",
        session=mock_session
    )

    await poller.begin_operation({"foo": "bar"})

    with pytest.raises(RuntimeError) as excinfo:
        # We haven't called wait_for_completion and status is not 'succeeded'
        poller.result()

    assert "No result available" in str(excinfo.value)

