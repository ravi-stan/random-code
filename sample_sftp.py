import asyncio
import asyncssh
import pandas as pd
import io
import logging
from datetime import datetime

async def sftp_read_csv_files_filtered_by_date(host, port, username, password, remote_folder):
    """
    Securely access an SFTP folder and read CSV files into memory that match today's date in the file name using asyncio.

    :param host: SFTP server hostname or IP address
    :param port: SFTP server port (default: 22)
    :param username: Username for SFTP login
    :param password: Password for SFTP login
    :param remote_folder: Path to the folder on the SFTP server
    :return: A dictionary with file names as keys and DataFrames as values
    """
    dataframes = {}
    try:
        # Establish the SSH connection
        async with asyncssh.connect(host, port=port, username=username, password=password) as conn:
            async with conn.start_sftp_client() as sftp:
                # List the contents of the remote folder
                files = await sftp.listdir(remote_folder)
                logging.info(f"Files in remote folder '{remote_folder}': {files}")

                # Get today's date in the required format
                today_date = datetime.now().strftime('%Y-%m-%d')
                logging.info(f"Filtering files with today's date: {today_date}")

                # Read each CSV file from the remote folder into memory if it matches today's date
                for file_name in files:
                    if file_name.lower().endswith('.csv') and today_date in file_name:
                        remote_path = f"{remote_folder}/{file_name}"
                        logging.info(f"Reading {remote_path} into memory")

                        # Open and read the remote file
                        async with sftp.open(remote_path, 'r') as remote_file:
                            file_content = await remote_file.read()

                            # Use StringIO to read the CSV content into pandas DataFrame
                            df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))

                            # Store the DataFrame in the dictionary
                            dataframes[file_name] = df

        logging.info("All matching CSV files read into memory successfully!")
        return dataframes

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

    dataframes = await sftp_read_csv_files_filtered_by_date(SFTP_HOST, SFTP_PORT, SFTP_USERNAME, SFTP_PASSWORD, REMOTE_FOLDER)

    # Now you can process the DataFrames as needed
    if dataframes:
        for file_name, df in dataframes.items():
            logging.info(f"Data from {file_name}:")
            print(df.head())

if __name__ == "__main__":
    asyncio.run(main())
