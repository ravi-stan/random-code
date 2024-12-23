import asyncio
import asyncssh
import pandas as pd
import io
import logging
from datetime import datetime
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock

def filter_files_by_date_and_prefix(files, today_date, prefixes):
    """
    Filter files by today's date and specified prefixes.

    :param files: List of file names
    :param today_date: Today's date in YYYY-MM-DD format
    :param prefixes: List of prefixes to filter files
    :return: List of tuples (prefix, file_name) matching the criteria
    """
    return [
        (prefix, file_name)
        for file_name in files
        if file_name.lower().endswith('.csv') and today_date in file_name
        for prefix in prefixes
        if file_name.startswith(prefix)
    ]

async def fetch_file_content(sftp, remote_path):
    """
    Fetch the content of a file from the SFTP server.

    :param sftp: SFTP client object
    :param remote_path: Path to the remote file
    :return: File content as a string
    """
    async with sftp.open(remote_path, 'r') as remote_file:
        return await remote_file.read()

async def list_remote_files(sftp, remote_folder):
    """
    List files in a remote folder.

    :param sftp: SFTP client object
    :param remote_folder: Path to the remote folder
    :return: List of file names
    """
    return await sftp.listdir(remote_folder)

async def process_filtered_files(sftp, remote_folder, filtered_files):
    """
    Process filtered files by reading their content and converting to DataFrames.

    :param sftp: SFTP client object
    :param remote_folder: Path to the remote folder
    :param filtered_files: List of tuples (prefix, file_name)
    :return: Dictionary of prefixes and concatenated DataFrames
    """
    dataframes = {}
    for prefix, file_name in filtered_files:
        remote_path = f"{remote_folder}/{file_name}"
        logging.info(f"Reading {remote_path} into memory")

        file_content = await fetch_file_content(sftp, remote_path)
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        if prefix not in dataframes:
            dataframes[prefix] = []
        dataframes[prefix].append(df)

    return {prefix: pd.concat(dfs) for prefix, dfs in dataframes.items() if dfs}

async def sftp_read_csv_files_filtered_by_date_and_prefixes(host, port, username, password, remote_folder, prefixes):
    """
    Securely access an SFTP folder and read CSV files into memory that match today's date and specific prefixes in the file name using asyncio.

    :param host: SFTP server hostname or IP address
    :param port: SFTP server port (default: 22)
    :param username: Username for SFTP login
    :param password: Password for SFTP login
    :param remote_folder: Path to the folder on the SFTP server
    :param prefixes: List of prefixes to filter files
    :return: A dictionary with prefixes as keys and DataFrames as values
    """
    try:
        async with asyncssh.connect(host, port=port, username=username, password=password) as conn:
            async with conn.start_sftp_client() as sftp:
                files = await list_remote_files(sftp, remote_folder)
                logging.info(f"Files in remote folder '{remote_folder}': {files}")

                today_date = datetime.now().strftime('%Y-%m-%d')
                logging.info(f"Filtering files with today's date: {today_date}")

                filtered_files = filter_files_by_date_and_prefix(files, today_date, prefixes)
                logging.info(f"Filtered files: {filtered_files}")

                return await process_filtered_files(sftp, remote_folder, filtered_files)

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return None

# Example usage
async def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # SFTP server details
    SFTP_HOST = 'example.com'
    SFTP_PORT = 22
    SFTP_USERNAME = 'your_username'
    SFTP_PASSWORD = 'your_password'
    REMOTE_FOLDER = '/path/to/remote/folder'
    PREFIXES = ['prefix1', 'prefix2', 'prefix3']

    dataframes = await sftp_read_csv_files_filtered_by_date_and_prefixes(SFTP_HOST, SFTP_PORT, SFTP_USERNAME, SFTP_PASSWORD, REMOTE_FOLDER, PREFIXES)

    # Now you can process the DataFrames as needed
    if dataframes:
        for prefix, df in dataframes.items():
            logging.info(f"Data for prefix {prefix}:")
            print(df.head())

if __name__ == "__main__":
    asyncio.run(main())

# Unit tests
class TestSFTPFunctions(IsolatedAsyncioTestCase):

    async def test_fetch_file_content(self):
        mock_sftp = MagicMock()
        mock_file = AsyncMock()
        mock_file.read.return_value = b"col1,col2\nval1,val2\n"
        mock_sftp.open.return_value = mock_file

        content = await fetch_file_content(mock_sftp, "/fake/path/to/file.csv")
        self.assertEqual(content, b"col1,col2\nval1,val2\n")
        mock_sftp.open.assert_called_once_with("/fake/path/to/file.csv", 'r')

    async def test_list_remote_files(self):
        mock_sftp = MagicMock()
        mock_sftp.listdir.return_value = ["prefix1_2024-12-20_file.csv", "prefix2_2024-12-20_file.csv"]

        files = await list_remote_files(mock_sftp, "/fake/remote/folder")
        self.assertEqual(files, ["prefix1_2024-12-20_file.csv", "prefix2_2024-12-20_file.csv"])
        mock_sftp.listdir.assert_called_once_with("/fake/remote/folder")

    def test_filter_files_by_date_and_prefix(self):
        files = ["prefix1_2024-12-20_file.csv", "prefix2_2024-12-20_file.csv", "other_2024-12-20_file.csv"]
        today_date = "2024-12-20"
        prefixes = ["prefix1", "prefix2"]

        filtered_files = filter_files_by_date_and_prefix(files, today_date, prefixes)
        expected = [("prefix1", "prefix1_2024-12-20_file.csv"), ("prefix2", "prefix2_2024-12-20_file.csv")]

        self.assertEqual(filtered_files, expected)

    async def test_process_filtered_files(self):
        mock_sftp = MagicMock()
        mock_sftp.open.return_value = AsyncMock(read=AsyncMock(return_value=b"col1,col2\nval1,val2\n"))
        filtered_files = [("prefix1", "prefix1_2024-12-20_file.csv")]

        result = await process_filtered_files(mock_sftp, "/fake/remote/folder", filtered_files)

        self.assertIn("prefix1", result)
        pd.testing.assert_frame_equal(
            result["prefix1"],
            pd.DataFrame({"col1": ["val1"], "col2": ["val2"]})
        )

    async def test_sftp_read_csv_files_filtered_by_date_and_prefixes(self):
        mock_sftp = MagicMock()
        mock_sftp.listdir.return_value = ["prefix1_2024-12-20_file.csv", "prefix2_2024-12-20_file.csv"]

        mock_sftp.open.return_value = AsyncMock(read=AsyncMock(return_value=b"col1,col2\nval1,val2\n"))

        with AsyncMock(asyncssh.connect) as mock_connect:
            mock_connect.return_value.start_sftp_client.return_value = mock_sftp

            result = await sftp_read_csv_files_filtered_by_date_and_prefixes(
                "host", 22, "user", "pass", "/fake/remote/folder", ["prefix1", "prefix2"]
            )

            self.assertIn("prefix1", result)
            self.assertIn("prefix2", result)
            pd.testing.assert_frame_equal(
                result["prefix1"],
                pd.DataFrame({"col1": ["val1"], "col2": ["val2"]})
            )
            pd.testing.assert_frame_equal(
                result["prefix2"],
                pd.DataFrame({"col1": ["val1"], "col2": ["val2"]})
            )
