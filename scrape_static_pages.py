import requests
from bs4 import BeautifulSoup
import json
import os
from playwright.sync_api import sync_playwright
#import json

pages = {
    "about_us": "https://qarba.com/about-us",
    "contact_us": "https://qarba.com/contact",
    "privacy_policy": "https://qarba.com/privacy-policy",
    "terms_conditions": "https://qarba.com/terms-conditions"
}

def scrape_with_playwright():
    data = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for name, url in pages.items():
            print(f"🔍 Scraping {name} from {url} ...")
            page.goto(url)
            page.wait_for_timeout(3000)  # wait 3s for JS to load
            content = page.inner_text("body")
            if content and len(content.strip()) > 100:
                data[name] = content.strip()
                print(f"✅ {name} content scraped ({len(content)} chars)")
            else:
                print(f"⚠️ No readable content found on {url}")
        browser.close()

    return data

scraped_data = scrape_with_playwright()

if scraped_data:
    with open("faqs.json", "r+", encoding="utf-8") as f:
        faqs = json.load(f)
        if "faqs" in faqs:
            faqs_list = faqs["faqs"]
        else:
            faqs_list = faqs
        for name, text in scraped_data.items():
            faqs_list.append({
                "question": f"What information can I find on the {name.replace('_', ' ')} page?",
                "answer": text[:1000]
            })
        f.seek(0)
        json.dump({"faqs": faqs_list}, f, indent=2, ensure_ascii=False)
        print("✅ Data added to faqs.json")
else:
    print("⚠️ No data scraped.")
