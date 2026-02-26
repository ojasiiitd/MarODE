import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


BASE_URL = "https://www.politifact.com/factchecks/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def scrape_article(url: str, session: requests.Session) -> List[str]:
    """
    Extract paragraph text from a PolitiFact fact-check page.
    """
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        article = soup.find("article", class_="m-textblock")
        if not article:
            return []

        paragraphs = [
            p.get_text(strip=True)
            for p in article.find_all("p")
            if p.get_text(strip=True)
        ]

        return paragraphs

    except requests.RequestException:
        return []


def extract_entry(row: Dict[str, str], session: requests.Session) -> Dict[str, Any] | None:
    claim = row.get("statement", "").strip()
    label = row.get("label", "").strip()
    raw_id = row.get("ID", "").replace(".json", "").strip()

    if not raw_id:
        return None

    url = f"{BASE_URL}{raw_id}/"
    evidence = scrape_article(url, session)

    if not evidence:
        return None

    return {
        "claim": claim,
        "label": label,
        "evidence_text": evidence,
    }


def process_liar_dataset(input_path: Path, output_path: Path, max_entries: int | None) -> None:
    results = []

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    with input_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")

        for i, line in enumerate(tqdm(f, desc="Processing LIAR")):
            fields = line.strip().split("\t")
            if len(fields) != len(header):
                continue

            row = dict(zip(header, fields))
            entry = extract_entry(row, session)

            if entry:
                results.append(entry)

            if max_entries and i >= max_entries:
                break

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LIAR dataset with scraped evidence."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-entries", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_liar_dataset(args.input, args.output, args.max_entries)