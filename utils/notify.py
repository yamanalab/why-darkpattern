import requests
from const.url import SLACK_WEBHOOK_URL


def notify_slack(text: str) -> None:
    requests.post(SLACK_WEBHOOK_URL, json={"text": text})
