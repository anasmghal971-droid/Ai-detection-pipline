import json
import time
import requests

class HFPushManager:
    def __init__(self, api_url):
        self.api_url = api_url

    def push(self, data):
        try:
            # Serialize data to JSON
            json_data = json.dumps(data)
            response = requests.post(self.api_url, json=json_data)

            # Raise an error for bad responses
            response.raise_for_status()

        except requests.exceptions.HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')  # Proper error handling
        except Exception as err:
            print(f'Other error occurred: {err}')  # Catch-all for other exceptions
        else:
            print('Push successful!')

    def backoff_logic(self, retries=5, initial_delay=1):
        for i in range(retries):
            try:
                return True  # Simulate success
            except Exception as e:
                if i < retries - 1:
                    time.sleep(initial_delay * (2 ** i))  # Exponential backoff
                else:
                    print(f'Failed after {retries} retries: {e}')  

# Usage Example
# manager = HFPushManager(api_url='https://example.com/push')
# manager.backoff_logic()
# manager.push({'key': 'value'})
