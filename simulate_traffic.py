import requests
import time
import random
from faker import Faker
from logger import get_logger

logger = get_logger(__name__)


fake = Faker()

BASE_URL = "http://127.0.0.1:5000"

HUMAN_PAGES = ["/", "/articles", "/about"]
BOT_PAGES   = ["/", "/articles", "/about", "/secret-data", "/secret-data", "/secret-data"]


def simulate_human(num_sessions=40):
    logger.info("Simulating human traffic...")

    for i in range(num_sessions):
        pages = random.sample(HUMAN_PAGES, k=random.randint(2, 3))

        for page in pages:
            try:
                requests.get(
                    BASE_URL + page,
                    headers={
                        "User-Agent"      : fake.user_agent(),
                        "Referer"         : "https://google.com",
                        "Accept-Language" : "en-US,en;q=0.9"
                    },
                    timeout=3
                )
            except Exception as e:
                logger.info(f"  Error: {e}")

            # reduced from 2–6s to 0.3–0.8s — still slower than bots
            time.sleep(random.uniform(0.3, 0.8))

        # reduced gap between sessions
        time.sleep(random.uniform(0.2, 0.5))

        # progress update every 10 sessions so you know it's running
        if (i + 1) % 10 == 0:
            logger.info(f"  {i+1}/{num_sessions} human sessions done...")

    logger.info(f"Done — {num_sessions} human sessions logged.")


def simulate_bot(num_sessions=40):
    logger.info("\nSimulating bot traffic...")

    bot_agents = [
        "python-requests/2.28.0",
        "Scrapy/2.11.0 (+https://scrapy.org)",
        "curl/7.88.1",
        "Mozilla/5.0 (compatible; Googlebot/2.1)",
        "Go-http-client/1.1"
    ]

    for i in range(num_sessions):
        for page in BOT_PAGES:
            try:
                requests.get(
                    BASE_URL + page,
                    headers={"User-Agent": random.choice(bot_agents)},
                    timeout=3
                )
            except Exception as e:
                logger.info(f"  Error: {e}")

            # bots are fast — barely any pause
            time.sleep(random.uniform(0.01, 0.05))

        time.sleep(random.uniform(0.05, 0.15))

        if (i + 1) % 10 == 0:
            logger.info(f"  {i+1}/{num_sessions} bot sessions done...")

    logger.info(f"Done — {num_sessions} bot sessions logged.")


def simulate_evasive_bot(num_sessions=20):
    logger.info("\nSimulating evasive bot traffic...")
    # Evasive bots try to mimic humans: normal user agents, realistic delays, mixing regular pages

    for i in range(num_sessions):
        # Mix of regular pages and secret pages
        pages = random.sample(HUMAN_PAGES, k=random.randint(1, 2)) + ["/secret-data"]
        random.shuffle(pages)

        for page in pages:
            try:
                requests.get(
                    BASE_URL + page,
                    headers={
                        "User-Agent"      : fake.user_agent(),
                        "Referer"         : "https://bing.com",
                        "Accept-Language" : "en-US,en;q=0.8"
                    },
                    timeout=3
                )
            except Exception as e:
                logger.info(f"  Error: {e}")

            # Evasive bots add human-like delays
            time.sleep(random.uniform(0.2, 0.7))

        time.sleep(random.uniform(0.1, 0.4))

        if (i + 1) % 5 == 0:
            logger.info(f"  {i+1}/{num_sessions} evasive bot sessions done...")

    logger.info(f"Done — {num_sessions} evasive bot sessions logged.")


if __name__ == "__main__":
    logger.info("Starting Victor traffic simulation...\n")
    simulate_human(num_sessions=40)
    simulate_bot(num_sessions=40)
    simulate_evasive_bot(num_sessions=20)
    logger.info("\nAll done! Traffic logged to data/victor_traffic.db")