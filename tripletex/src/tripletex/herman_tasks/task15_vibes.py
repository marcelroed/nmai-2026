import requests
import requests.exceptions
import rich

from tripletex.herman_tasks.utils import (
    TripletexCredentials,
    get_current_year_month_day_utc,
    get_customer_by_org_number,
    get_employee,
)
from tripletex.log_analysis._3_find_by_task import get_task

tasks, raw_data = get_task("Task 15", only_log_version_2=True)
# +
task = tasks[0]
log_data = raw_data[task.request_id][::-1]
# +
task.prompt

# +


(project,) = [
    x for x in log_data if x["message"].endswith("got tripletex data for /project")
]
project["extra"]["data"]["values"]
