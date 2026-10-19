
from pathlib import Path
import json
import time

import requests
from bs4 import BeautifulSoup

PROJECT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_DIR / "medbot_phase3_rag" / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = [
    {
        "source": "NIH MedlinePlus",
        "url": "https://medlineplus.gov/diabetes.html",
    },
    {
        "source": "NIH MedlinePlus",
        "url": "https://medlineplus.gov/highbloodpressure.html",
    },
    {
        "source": "CDC",
        "url": "https://www.cdc.gov/diabetes/about/index.html",
    },
    {
        "source": "WHO",
        "url": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
    },
    
    {
        "source": "NIH MedlinePlus",
        "url": "https://medlineplus.gov/asthma.html",
    },

]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36 "
        "MedBotAcademicProject/1.0"
    )
}

REMOVE_SELECTORS = [
    "script", "style", "nav", "footer", "header",
    "aside", "form", "button", "noscript",
    "[role='navigation']", "[role='complementary']",
    ".breadcrumb", ".breadcrumbs", ".related-links",
    ".related", ".social-share", ".share-tools",
    ".page-navigation", ".menu", ".sidebar",
]


def extract_text(html, source):
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else "Untitled"

    for selector in REMOVE_SELECTORS:
        for element in soup.select(selector):
            element.decompose()

    # Choose the most appropriate article container for each site.
    if "medlineplus.gov" in source:
        main = (
            soup.select_one("#maincontent")
            or soup.select_one("main")
            or soup.find("article")
        )
    elif "cdc.gov" in source:
        main = (
            soup.select_one("main")
            or soup.select_one("#content")
            or soup.find("article")
        )
    elif "who.int" in source:
        main = (
            soup.select_one("main")
            or soup.select_one("article")
            or soup.select_one("[role='main']")
        )
    else:
        main = soup.find("main") or soup.find("article")

    if main is None:
        return title, ""

    # Remove common related-content and navigation blocks inside main.
    for selector in [
        "nav", "aside", ".related", ".related-links",
        ".breadcrumb", ".share-tools", ".social-share",
    ]:
        for element in main.select(selector):
            element.decompose()

    lines = []

    # Extract headings and paragraphs, not every list item.
    # Lists are included only when they appear to be article content.
    for element in main.find_all(["h1", "h2", "h3", "p", "li"]):
        if element.find_parent(["nav", "aside"]):
            continue

        text = element.get_text(" ", strip=True)
        if not text:
            continue

        # Avoid obvious navigation/link-only entries.
        if element.name == "li":
            links = element.find_all("a")
            if links and len(links) == 1:
                link_text = links[0].get_text(" ", strip=True)
                if link_text and link_text.strip() == text.strip():
                    continue

        if text not in lines:
            lines.append(text)

    return title, "\n".join(lines)


def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    manifest = []
    success_count = 0

    for index, item in enumerate(SOURCES, start=1):
        url = item["url"]
        source = item["source"]

        print(f"\n[{index}/{len(SOURCES)}] Downloading: {url}")

        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "html" not in content_type.lower():
                raise ValueError(
                    f"Unexpected content type: {content_type}"
                )

            title, text = extract_text(response.text, url)

            if len(text.split()) < 100:
                raise ValueError(
                    "Extracted article text is too short. "
                    "The page may have changed or blocked access."
                )

            document = {
                "title": title,
                "source": source,
                "url": url,
                "collected_at": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.gmtime()
                ) + " UTC",
                "text": text,
            }

            filename = f"document_{index:02d}.json"
            output_path = DOCUMENTS_DIR / filename

            with output_path.open("w", encoding="utf-8") as file:
                json.dump(
                    document, file, indent=2, ensure_ascii=False
                )

            manifest.append({
                "file": filename,
                "title": title,
                "source": source,
                "url": url,
                "word_count": len(text.split()),
                "preview": text[:350],
                "status": "success",
            })

            success_count += 1

            print(f"Saved: {output_path}")
            print(f"Title: {title}")
            print(f"Words extracted: {len(text.split())}")
            print(f"Preview: {text[:250].replace(chr(10), ' ')}")

        except Exception as exc:
            manifest.append({
                "source": source,
                "url": url,
                "status": "failed",
                "error": str(exc),
            })
            print(f"Download failed: {exc}")

    manifest_path = DOCUMENTS_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print(
        f"\nCollection finished: "
        f"{success_count}/{len(SOURCES)} pages saved."
    )
    print(f"Manifest: {manifest_path}")
    print(
        "\nReview the previews and confirm they contain medical "
        "article content before rebuilding the vector index."
    )


if __name__ == "__main__":
    main()
