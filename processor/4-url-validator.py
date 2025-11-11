import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time
import re

# ==== CONFIG ====
URLS_FILE = "/Users/user/Documents/AI/parallel-process/processor/url.txt"  # File containing URLs (one per line)
OUTPUT_FILE = "url_validation_report.csv"
MAX_WORKERS = 30  # Number of concurrent threads
TIMEOUT = 15  # Timeout in seconds for each request
RETRY_ATTEMPTS = 2  # Number of retry attempts for failed URLs

# ==== STEP 1: Load and Clean URLs ====
def load_urls_from_file(filename):
    """Load URLs from a text file and clean them"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract URLs using regex pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]\n]+'
        urls = re.findall(url_pattern, content)
        
        # Clean URLs - remove any trailing characters
        cleaned_urls = []
        for url in urls:
            # Remove trailing punctuation that might not be part of URL
            url = url.rstrip('.,;:)')
            # Remove bullet points or numbering
            url = re.sub(r'^[\d]+\.\s*', '', url)
            url = re.sub(r'^[•\-]\s*', '', url)
            if url:
                cleaned_urls.append(url.strip())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in cleaned_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        return unique_urls
    except FileNotFoundError:
        print(f"⚠️  File '{filename}' not found.")
        return []
    except Exception as e:
        print(f"⚠️  Error reading file: {e}")
        return []

# ==== STEP 2: Validate Single URL ====
def validate_url(url, retry_count=0):
    """
    Validate a single URL and return status information
    """
    result = {
        'url': url,
        'status': 'Unknown',
        'status_code': None,
        'response_time': None,
        'content_type': None,
        'content_length': None,
        'error': None,
        'final_url': None
    }
    
    try:
        start_time = time.time()
        # Use HEAD request first for efficiency
        response = requests.head(url, timeout=TIMEOUT, allow_redirects=True)
        response_time = time.time() - start_time
        
        result['status_code'] = response.status_code
        result['response_time'] = round(response_time, 2)
        result['content_type'] = response.headers.get('Content-Type', 'Unknown')
        result['content_length'] = response.headers.get('Content-Length', 'Unknown')
        result['final_url'] = response.url if response.url != url else None
        
        if response.status_code == 200:
            result['status'] = 'Valid'
        elif response.status_code == 404:
            result['status'] = 'Not Found'
        elif response.status_code == 403:
            result['status'] = 'Forbidden'
        elif response.status_code >= 500:
            result['status'] = 'Server Error'
        elif response.status_code >= 400:
            result['status'] = 'Client Error'
        elif response.status_code >= 300:
            result['status'] = 'Redirect'
        else:
            result['status'] = 'Other'
            
    except requests.exceptions.Timeout:
        result['status'] = 'Timeout'
        result['error'] = f'Request timeout after {TIMEOUT}s'
    except requests.exceptions.SSLError as e:
        result['status'] = 'SSL Error'
        result['error'] = str(e)[:150]
    except requests.exceptions.ConnectionError as e:
        result['status'] = 'Connection Error'
        result['error'] = str(e)[:150]
    except requests.exceptions.TooManyRedirects:
        result['status'] = 'Too Many Redirects'
        result['error'] = 'Exceeded maximum redirects'
    except requests.exceptions.RequestException as e:
        result['status'] = 'Request Error'
        result['error'] = str(e)[:150]
    except Exception as e:
        result['status'] = 'Unknown Error'
        result['error'] = str(e)[:150]
    
    # Retry logic for failed requests
    if result['status'] in ['Timeout', 'Connection Error', 'Server Error'] and retry_count < RETRY_ATTEMPTS:
        time.sleep(1)
        return validate_url(url, retry_count + 1)
    
    return result

# ==== STEP 3: Validate URLs Concurrently ====
def validate_urls_concurrent(urls):
    """
    Validate multiple URLs concurrently using ThreadPoolExecutor
    """
    results = []
    total = len(urls)
    
    print(f"🔍 Starting validation of {total} URLs...\n")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(validate_url, url): url for url in urls}
        
        completed = 0
        for future in as_completed(future_to_url):
            completed += 1
            result = future.result()
            results.append(result)
            
            # Print progress
            status_icon = {
                'Valid': '✅',
                'Not Found': '❌',
                'Forbidden': '🚫',
                'Server Error': '🔥',
                'Client Error': '⚠️',
                'Error': '⚠️',
                'Timeout': '⏱️',
                'Connection Error': '🔌',
                'SSL Error': '🔒',
                'Redirect': '↪️',
                'Other': '❓'
            }.get(result['status'], '•')
            
            # Show first 60 chars of URL
            url_display = result['url'][:60] + '...' if len(result['url']) > 60 else result['url']
            status_display = f"{result['status']}"
            if result['status_code']:
                status_display += f" ({result['status_code']})"
            
            print(f"[{completed}/{total}] {status_icon} {status_display}: {url_display}")
    
    return results

# ==== STEP 4: Generate Detailed Report ====
def print_report(df, total_time):
    """Print detailed validation report"""
    print("\n" + "=" * 80)
    print(" VALIDATION REPORT ".center(80, "="))
    print("=" * 80)
    
    print(f"\n📊 SUMMARY:")
    print(f"   • Total URLs validated: {len(df)}")
    print(f"   • Unique URLs: {df['url'].nunique()}")
    print(f"   • Total time taken: {round(total_time, 2)}s")
    print(f"   • Average time per URL: {round(total_time / len(df), 2)}s")
    
    print(f"\n📈 STATUS BREAKDOWN:")
    status_counts = df['status'].value_counts()
    for status, count in status_counts.items():
        percentage = (count / len(df)) * 100
        icon = {
            'Valid': '✅',
            'Not Found': '❌',
            'Forbidden': '🚫',
            'Server Error': '🔥',
            'Client Error': '⚠️',
            'Timeout': '⏱️',
            'Connection Error': '🔌',
            'SSL Error': '🔒',
            'Redirect': '↪️',
            'Other': '❓'
        }.get(status, '•')
        print(f"   {icon} {status}: {count} ({percentage:.1f}%)")
    
    # Response time statistics for valid URLs
    valid_df = df[df['status'] == 'Valid']
    if len(valid_df) > 0:
        print(f"\n⚡ RESPONSE TIME STATISTICS (Valid URLs):")
        print(f"   • Fastest: {valid_df['response_time'].min()}s")
        print(f"   • Slowest: {valid_df['response_time'].max()}s")
        print(f"   • Average: {round(valid_df['response_time'].mean(), 2)}s")
        print(f"   • Median: {round(valid_df['response_time'].median(), 2)}s")
    
    # Show invalid URLs
    invalid_df = df[df['status'] != 'Valid']
    if len(invalid_df) > 0:
        print(f"\n⚠️  INVALID/FAILED URLs ({len(invalid_df)}):")
        print("=" * 80)
        for idx, row in invalid_df.head(20).iterrows():
            url_display = row['url'][:65] + '...' if len(row['url']) > 65 else row['url']
            error_msg = f" - {row['error'][:50]}" if pd.notna(row['error']) else ""
            status_display = f"[{row['status']}"
            if pd.notna(row['status_code']):
                status_display += f" {row['status_code']}"
            status_display += "]"
            print(f"   {status_display} {url_display}{error_msg}")
        
        if len(invalid_df) > 20:
            print(f"\n   ... and {len(invalid_df) - 20} more invalid URLs")
        print("=" * 80)

# ==== MAIN EXECUTION ====
if __name__ == "__main__":
    print("=" * 80)
    print(" URL VALIDATOR ".center(80, "="))
    print("=" * 80 + "\n")
    
    # Load URLs
    urls = load_urls_from_file(URLS_FILE)
    
    if not urls:
        print("❌ No URLs found. Please check your input file.")
        exit(1)
    
    print(f"📋 Loaded {len(urls)} unique URLs from file")
    print(f"⚙️  Max concurrent workers: {MAX_WORKERS}")
    print(f"⏱️  Timeout per request: {TIMEOUT}s")
    print(f"🔄 Retry attempts: {RETRY_ATTEMPTS}\n")
    
    # Validate URLs
    start_time = time.time()
    results = validate_urls_concurrent(urls)
    total_time = time.time() - start_time
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by status (Valid first, then others)
    status_order = ['Valid', 'Redirect', 'Not Found', 'Forbidden', 'Client Error', 
                    'Server Error', 'Timeout', 'Connection Error', 'SSL Error', 
                    'Request Error', 'Other', 'Unknown Error']
    df['status_order'] = df['status'].apply(lambda x: status_order.index(x) if x in status_order else 999)
    df = df.sort_values('status_order').drop('status_order', axis=1)
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Generate report
    print_report(df, total_time)
    
    print(f"\n💾 OUTPUT:")
    print(f"   • Full report saved to: {OUTPUT_FILE}")
    
    print("\n" + "=" * 80)
    print(" VALIDATION COMPLETED ".center(80, "="))
    print("=" * 80 + "\n")