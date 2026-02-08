#!/usr/bin/env python3
"""
Web scraper for target URLs using Playwright.
Scrapes HTML content from URLs listed in frequent_targets.json and saves
individual JSON files with scraped content.
"""

import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from playwright.sync_api import sync_playwright, Page, Browser
import time


class TargetScraper:
    """Scrapes target URLs and saves HTML content to JSON files."""
    
    def __init__(self, input_file: str, output_dir: str = "./data/targets/page-content"):
        """
        Initialize the scraper.
        
        Args:
            input_file: Path to frequent_targets.json
            output_dir: Directory to save scraped content
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.data = []
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_targets(self) -> None:
        """Load target data from JSON file."""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} target records")
    
    def get_url_sha256(self, url: str) -> str:
        """
        Generate SHA256 hash of a URL.
        
        Args:
            url: The URL to hash
            
        Returns:
            SHA256 hash as hex string
        """
        return hashlib.sha256(url.encode('utf-8')).hexdigest()
    
    def deduplicate_urls(self, record: Dict) -> List[str]:
        """
        Combine website_url and top_10_domain_urls, then deduplicate.
        
        Args:
            record: A record from frequent_targets.json
            
        Returns:
            Deduplicated list of URLs
        """
        urls = []
        
        # Add website_url if it exists
        if 'website_url' in record and record['website_url']:
            urls.append(record['website_url'])
        
        # Add top_10_domain_urls if it exists
        if 'top_10_domain_urls' in record and record['top_10_domain_urls']:
            urls.extend(record['top_10_domain_urls'])
        
        # Deduplicate while preserving order
        seen = set()
        deduplicated = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                deduplicated.append(url)
        
        return deduplicated
    
    def scrape_page(self, page: Page, url: str) -> Dict:
        """
        Scrape a single page and return structured data.
        
        Args:
            page: Playwright page object
            url: URL to scrape
            
        Returns:
            Dictionary with scraped data
        """
        try:
            # Navigate to the page with timeout
            page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait a bit for any dynamic content
            time.sleep(2)
            
            # Extract data
            html_content = page.content()
            html_text = page.inner_text('body')
            title = page.title()
            
            # Get current timestamp
            scrape_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculate SHA256 of URL
            url_hash = self.get_url_sha256(url)
            
            result = {
                'url': url,
                'sha256': url_hash,
                'html_content': html_content,
                'html_text': html_text,
                'title': title,
                'label': 'benign',
                'scrape_time': scrape_time,
                'status': 'success'
            }
            
            return result
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            scrape_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            url_hash = self.get_url_sha256(url)
            
            return {
                'url': url,
                'sha256': url_hash,
                'html_content': '',
                'html_text': '',
                'title': '',
                'label': 'benign',
                'scrape_time': scrape_time,
                'status': f'error: {str(e)}'
            }
    
    def save_scraped_data(self, data: Dict) -> bool:
        """
        Save scraped data to a JSON file.
        
        Args:
            data: Dictionary containing scraped data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filename = f"{data['sha256']}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"Saved: {filename} ({data['url']})")
            return True
            
        except Exception as e:
            print(f"Error saving {data['url']}: {str(e)}")
            return False
    
    def scrape_all_targets(self) -> None:
        """Scrape all target URLs using Playwright with headed browser."""
        # Track scraping status for each record
        scraping_statuses = []
        
        with sync_playwright() as p:
            # Launch browser in headless mode
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = context.new_page()
            
            # Process each record
            for idx, record in enumerate(self.data):
                print(f"\n{'='*80}")
                print(f"Processing record {idx + 1}/{len(self.data)}: {record.get('target', 'Unknown')}")
                print(f"{'='*80}")
                
                # Get deduplicated URLs for this record
                urls = self.deduplicate_urls(record)
                print(f"Found {len(urls)} unique URLs to scrape")
                
                # Track status for this record
                record_status = []
                
                # Scrape each URL
                for url_idx, url in enumerate(urls):
                    print(f"\n[{url_idx + 1}/{len(urls)}] Scraping: {url}")
                    
                    # Scrape the page
                    scraped_data = self.scrape_page(page, url)
                    
                    # Save to file
                    save_success = self.save_scraped_data(scraped_data)
                    
                    # Track status
                    status = scraped_data['status'] if save_success else 'save_failed'
                    record_status.append((url, status))
                
                # Store scraping status for this record
                record['scraping_status'] = record_status
                scraping_statuses.append(record_status)
            
            # Close browser
            page.close()
            context.close()
            browser.close()
        
        print(f"\n{'='*80}")
        print("Scraping completed!")
        print(f"{'='*80}")
    
    def update_source_file(self) -> None:
        """Update the source JSON file with scraping_status field."""
        try:
            with open(self.input_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            print(f"\nUpdated {self.input_file} with scraping status")
        except Exception as e:
            print(f"Error updating source file: {str(e)}")
    
    def run(self) -> None:
        """Execute the complete scraping workflow."""
        print("Starting Target URL Scraper")
        print("="*80)
        
        # Load targets
        self.load_targets()
        
        # Scrape all URLs
        self.scrape_all_targets()
        
        # Update source file with status
        self.update_source_file()
        
        print("\n✓ All done!")


def main():
    """Main entry point."""
    input_file = "./cleaned/frequent_targets.json"
    output_dir = "./data/targets"
    
    scraper = TargetScraper(input_file, output_dir)
    scraper.run()


if __name__ == "__main__":
    main()
