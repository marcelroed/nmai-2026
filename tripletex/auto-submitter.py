import random
import time

import requests

n_submitted = 0
n_to_submit = 25
for i in range(n_to_submit * 10):
    if n_submitted >= n_to_submit:
        print("DONE! bye")
        break
    print(f"submitting {n_submitted=}/{n_to_submit} ; {i=}")
    cookies = {
        "access_token": "",  # removed
    }

    submissions_data = requests.get(
        "https://api.ainm.no/tripletex/my/submissions", cookies=cookies
    ).json()

    auth_token = submissions_data[0]["id"]

    json_data = {
        "endpoint_url": "https://tripletex-agent-1010470531441.europe-north1.run.app/solve",
        "endpoint_api_key": auth_token,
    }
    to_sleep = random.randint(3, 5) + random.random()

    current_in_flight = [
        x["status"]
        for x in submissions_data
        if x["status"] != "completed" and x["status"] != "failed"
    ]
    print(f"{current_in_flight=}")

    if len(current_in_flight) >= 3:
        print("waiting another round")
        print(f"sleeping for {to_sleep}")
        time.sleep(to_sleep)
        print()
        continue

    n_submitted += 1
    print(f"\x1b[32mmaking request\x1b[0m {json_data=}")
    response = requests.post(
        "https://api.ainm.no/tasks/cccccccc-cccc-cccc-cccc-cccccccccccc/submissions",
        cookies=cookies,
        json=json_data,
    )
    print(response.json())
    print(f"sleeping for {to_sleep}")
    time.sleep(to_sleep)
    print()
