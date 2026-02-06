import argparse
from pathlib import Path
from pyparsing import Any
from tqdm import tqdm
import json
import multiprocessing as mp
import random
from bs4 import BeautifulSoup
import re
import sys
from google.oauth2 import service_account
from google import genai
from pydantic import BaseModel
import time
from datetime import datetime
import base64

class RateLimiter:
    """Shared rate limiter for multiprocessing pool"""
    def __init__(self, manager, tpm_limit, rpm_limit):
        self.namespace = manager.Namespace()
        self.lock = manager.Lock()
        
        # Initialize shared state
        self.namespace.tokens_used = 0
        self.namespace.requests_made = 0
        self.namespace.window_start = time.time()
        
        # Set limits
        self.tpm_limit = tpm_limit
        self.rpm_limit = rpm_limit
    
    def wait_if_needed(self, estimated_tokens=0):
        """Check if limits are exceeded and wait if necessary"""
        while True:
            should_wait = False
            wait_time = 0
            
            with self.lock:
                current_time = time.time()
                elapsed = current_time - self.namespace.window_start
                
                # Reset window if more than 60 seconds have passed
                if elapsed >= 60:
                    self.namespace.tokens_used = 0
                    self.namespace.requests_made = 0
                    self.namespace.window_start = current_time
                
                # Check if adding this request would exceed limits
                would_exceed_tpm = (self.namespace.tokens_used + estimated_tokens) > self.tpm_limit
                would_exceed_rpm = (self.namespace.requests_made + 1) > self.rpm_limit
                
                if would_exceed_tpm or would_exceed_rpm:
                    # Calculate how long to wait until the window resets
                    should_wait = True
                    wait_time = 61 - elapsed
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Rate limit reached. "
                          f"Tokens: {self.namespace.tokens_used}/{self.tpm_limit}, "
                          f"Requests: {self.namespace.requests_made}/{self.rpm_limit}. "
                          f"Waiting {wait_time:.1f}s...")
                else:
                    # Limits OK - reserve our spot by pre-incrementing
                    self.namespace.tokens_used += estimated_tokens
                    self.namespace.requests_made += 1
                    return  # Exit the loop and proceed
            
            # Sleep OUTSIDE the lock to avoid blocking other processes
            if should_wait:
                time.sleep(wait_time)
                # After sleep, loop back to re-check limits in case other processes used quota
    
    def adjust_usage(self, estimated_tokens, actual_tokens):
        """Adjust token count after API call if estimate was wrong"""
        with self.lock:
            # Remove estimated tokens and add actual tokens
            self.namespace.tokens_used = self.namespace.tokens_used - estimated_tokens + actual_tokens


def count_tokens(contents, model, location, credentials):
    client = genai.Client(
        vertexai=True, 
        project=credentials.project_id, 
        location=location,
        credentials=credentials
    )
    n_tokens = client.models.count_tokens(model=model, contents=contents)
    return n_tokens.total_tokens


def SimplifyHTML(inputHTML, args, credentials):
    soup = BeautifulSoup(inputHTML, "html.parser")

    processedHTML = str(soup)

    if count_tokens(
        processedHTML, model=args.model, location=args.location, credentials=credentials
    ) < args.max_html_tokens:
        return processedHTML, '0-as-is'
    
    for tag in soup(["style", "script"]):
        tag.decompose()

    for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.startswith("<!--")):
        comment.extract()

    processedHTML = str(soup)

    if count_tokens(
        processedHTML, model=args.model, location=args.location, credentials=credentials
    ) < args.max_html_tokens:
        return processedHTML, '1-decomposed-style-script-comments'

    important_tags = {"p", "a", "img", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li"}
    for tag in soup.find_all(True):
        if tag.name not in important_tags:
            tag.unwrap()

    for tag in soup.find_all(True):
        if not tag.get_text(strip=True):
            tag.decompose()

    for tag in soup.find_all("a", href=True):
        tag["href"] = re.sub(r'^(https?://)?(www\.)?', '', tag["href"])

    for tag in soup.find_all("img", src=True):
        tag["src"] = re.sub(r'^(https?://)?(www\.)?', '', tag["src"])

    processedHTML = str(soup)

    if count_tokens(
        processedHTML, model=args.model, location=args.location, credentials=credentials
    ) < args.max_html_tokens:
        return processedHTML, '2-unwrapped-nonimportant-tags'

    while count_tokens(
        processedHTML, model=args.model, location=args.location, credentials=credentials
    ) > args.max_html_tokens:
        tags = soup.find_all(True)
        if not tags:
            break
        midpoint = len(tags) // 2
        tags[midpoint].decompose()
        processedHTML = str(soup)

    return processedHTML, '3-decomposed-mid-tags'


def generate_gemini_prompt(url: str, html: str) -> str:
    """
    Generates a prompt for Gemini API to analyze a URL and HTML for phishing.

    Args:
        url: The URL of the webpage.
        html: The HTML of the webpage.

    Returns:
        A string containing the Gemini API prompt.
    """

    prompt = f"""
        You are an expert in analyzing URL and multi-lingual HTML to classify webpage as phishing or legitimate page.
        Focus only on identifying if the page is phishing and its corresponding risk score.
        Given the URL and HTML, perform the following analysis for any Social Engineering techniques often used in phishing attacks:
        First perform thorough REASONING on the URL and HTML looking for suspicious elements or inconsistencies, domain issues, detect content red flags, evaluate security indicators and inconsistencies, and then classify with a confidence score.
        1.  **PHISHING:** Whether the webpage is a phishing page or a legitimate page. (True if phishing, False if legitimate)
        2.  **SCORE:** Indicates phishing risk on a scale of 0.0 to 1.0 (inclusive), where 0.0 is the least likely to have phishing content and 1.0 is the most likely.
        3.  **REASON:** Briefly (one sentence) explain the reasoning behind your determination.
        Provide the extracted information in JSON format with the following keys:
        - phishing: boolean (whether the site is a phishing site or a legitimate site)
        - score: float (indicates phishing risk on a scale of 0.0 to 1.0)
        - reason: str (one sentence reason)

        URL: ```{url}```
        HTML: ```{html}```
    """
    # prompt = f"""
    #     You are an expert in analyzing URL and multi-lingual HTML to classify webpage as phishing or legitimate page.
    #     Focus only on identifying if the page is phishing and its corresponding risk score.
    #     Given the base64 encoded URL and base64 encoded HTML, perform the following analysis after decoding the URL and HTML for any Social Engineering techniques often used in phishing attacks:
    #     First perform thorough REASONING on the URL and HTML looking for suspicious elements or inconsistencies, domain issues, detect content red flags, evaluate security indicators and inconsistencies, and then classify with a confidence score.
    #     1.  **PHISHING:** Whether the webpage is a phishing page or a legitimate page. (True if phishing, False if legitimate)
    #     2.  **SCORE:** Indicates phishing risk on a scale of 0.0 to 1.0 (inclusive), where 0.0 is the least likely to have phishing content and 1.0 is the most likely.
    #     3.  **REASON:** Briefly (one sentence) explain the reasoning behind your determination.
    #     Provide the extracted information in JSON format with the following keys:
    #     - phishing: boolean (whether the site is a phishing site or a legitimate site)
    #     - score: float (indicates phishing risk on a scale of 0.0 to 1.0)
    #     - reason: str (one sentence reason)

    #     URL: ```{url}```
    #     HTML: ```{html}```
    # """
    return prompt

def prompt_n_listen(contents: str, args: argparse.Namespace, credentials: Any) -> str:
    # Define response schema
    class PhishingResponse(BaseModel):
        phishing: bool
        score: float
        reason: str

    try:
        safety_settings = [
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_IMAGE_HATE,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_IMAGE_HARASSMENT,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_JAILBREAK,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]
        # Initialize Gemini client
        client = genai.Client(
            vertexai=True, 
            project=credentials.project_id, 
            location=args.location,
            credentials=credentials,
            http_options=genai.types.HttpOptions(
                api_version="v1",
            )
        )
        # Generate response
        response = client.models.generate_content(
            model=args.model,
            contents=contents,
            config=genai.types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=PhishingResponse,
                safety_settings=safety_settings,
            ),
        )
        # Parse response
        response_text = (
            response.text.decode('utf-8') 
            if isinstance(response.text, bytes) else response.text
        )
        # Convert response to JSON
        json_output = json.loads(response_text)
        
        # Return relevant fields
        return {
            "phishing": json_output["phishing"],
            "score": json_output["score"],
            "reason": json_output["reason"],
            "exception": None
        }
    except Exception as e:
        # Handle exceptions and return failure response
        print(f"Exception during Gemini API call: {e}")
        return {
            "phishing": False,
            "score": -1.0,
            "reason": "Failed",
            "exception": str(e)
        }

def predict(params):
    filename, args, credentials, rate_limiter, pred_dir = params
    # print(f"Processing: {filename.stem}")
    # Read input JSON
    with open(filename, 'r') as f:
        data = json.load(f)

    sha256 = data["sha256"]
    label = 1 if data["label"] == "phish" else 0
    
    # Shrink HTML if needed
    processed_html, simplification = SimplifyHTML(data["html"], args, credentials) if data["html"] else ("", "")

    # Generate Gemini prompt
    # encoded_html = base64.b64encode(processed_html.encode('utf-8')).decode('utf-8') if processed_html else ""
    # encoded_url = base64.b64encode(data["url"].encode('utf-8')).decode('utf-8') if data["url"] else ""
    contents = generate_gemini_prompt(data['url'], processed_html)

    # Count tokens for rate limiting
    try:
        estimated_tokens_content = count_tokens(contents, model=args.model, location=args.location, credentials=credentials)
        # estimated_tokens = count_tokens(encoded_html, model=args.model, location=args.location, credentials=credentials)
        print(f"Estimated tokens for {filename.stem}: Prompt={estimated_tokens_content}, Encoded HTML={estimated_tokens}")
    except:
        # If token counting fails, use a conservative estimate
        estimated_tokens = len(contents) // 4  # Rough estimate: 1 token per 4 chars
        # estimated_tokens = len(encoded_html) // 4
    
    # Wait if rate limits would be exceeded (also reserves our quota)
    rate_limiter.wait_if_needed(estimated_tokens)

    # Get prediction from Gemini
    result = {
        "sha256": sha256,
        "url": data["url"],
        "label": label,
        "simplification": simplification,
        **prompt_n_listen(contents, args, credentials)
    }

    if result["exception"] is None:
        # Save successful prediction
        pred_file = pred_dir / f"{sha256}.json"
        with open(pred_file, 'w') as f:
            json.dump(result, f)
    
    # Note: We pre-incremented in wait_if_needed, so no need to update again
    # If we wanted to adjust for actual vs estimated tokens, we could do:
    # rate_limiter.adjust_usage(estimated_tokens, actual_tokens)
    # print(f"Processed {filename.stem}: {result}")
    return result




def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Predict using Gemini LLM model.")
    parser.add_argument("--test_dir", type=Path, required=True, help="Directory containing test data.")
    parser.add_argument("--split", type=str, default="test", help="Data split to use (default: test).")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to process.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling test files.")
    parser.add_argument("--num_procs", type=int, default=4, help="Number of parallel processes to use.")
    parser.add_argument("--service_account_path", required=True, help="Path to the service account JSON file")
    parser.add_argument("--location", default="us-central1", help="Location for the Gemini model")
    parser.add_argument("--model", required=True, help="Gemini model name (e.g., 'gemini-1.5-flash')")
    parser.add_argument("--max_html_tokens", type=int, default=1_000_000, help="Maximum tokens for the model response.")
    parser.add_argument("--pred_dir", type=Path, default="./data/predictions_json/llm/gemini", help="Path to save predictions.")
    parser.add_argument("--tpm_limit", type=int, default=2_000_000, help="Tokens per minute limit (default: 2M for Gemini 1.5 Flash).")
    parser.add_argument("--rpm_limit", type=int, default=1000, help="Requests per minute limit (default: 1000 for Gemini 1.5 Flash).")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for failed predictions.")
    parser.add_argument("--retry_delay", type=int, default=300, help="Delay in seconds before retrying failed predictions.")

    args = parser.parse_args()  
    
    # Validate test directory
    assert args.test_dir.is_dir(), f"The path {args.test_dir} is not a valid directory."

    # Create prediction directory if it doesn't exist
    pred_dir = args.pred_dir / args.split

    # Collect sha256's of successful predictions to avoid repeated LLM call
    # But save the successful predictions to be concatenated with new predictions and written back
    predicted_sha256s = []
    if pred_dir.exists():
        predicted_sha256s = [f.stem for f in pred_dir.glob("*.json")]
    else:
        pred_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(predicted_sha256s)} successful predictions already present.")
        

    # Get list of test files
    phish_test_files = list((args.test_dir / "phishing").glob("*.json"))
    print(f"Found {len(phish_test_files)} phishing test files in {args.test_dir}.")
    benign_test_files = list((args.test_dir / "benign").glob("*.json"))
    print(f"Found {len(benign_test_files)} benign test files in {args.test_dir}.")
    ratio = len(phish_test_files) / (len(phish_test_files) + len(benign_test_files))
    print(f"Phishing to benign ratio: {ratio:.2f}")

    if args.n_samples > 0:
        random.seed(args.seed)
        phish_test_files = random.sample(phish_test_files, int(args.n_samples * ratio))
        benign_test_files = random.sample(benign_test_files, (args.n_samples - len(phish_test_files)))
        print(f"Sampled {len(phish_test_files)} phishing test files and {len(benign_test_files)} benign test files.")

    test_files = phish_test_files + benign_test_files
    print(f"Using {len(test_files)} test files for prediction.")

    # Filter out files with existing successful predictions
    if predicted_sha256s:
        test_files = [f for f in test_files if f.stem not in predicted_sha256s]
        print(f"{len(test_files)} files remaining after filtering existing successful predictions.")
        print(test_files)
    

    # Set up credentials
    credentials = service_account.Credentials.from_service_account_file(
        args.service_account_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    # Create shared rate limiter
    tpm = int(args.tpm_limit * 0.8)
    rpm = int(args.rpm_limit * 0.99)
    
    # Prepare parameters and get predictions from LLM
    with mp.Manager() as manager:
        rate_limiter = RateLimiter(manager, tpm_limit=tpm, rpm_limit=rpm)
        params = [(f, args, credentials, rate_limiter, pred_dir) for f in test_files]
        print(f"Rate limiter initialized: TPM={tpm}, RPM={rpm}")
        
        with mp.Pool(args.num_procs) as pool:
            results = pool.map(predict, tqdm(params, total=len(params), desc="Predicting"))
    
    predicted_sha256s.extend([r['sha256'] for r in results if r.get("exception") is None])
    
    # Retry failed predictions
    failed_sha256s = [r['sha256'] for r in results if r.get("exception") is not None]
    retry_count = 0
    all_results = results.copy()  # Keep track of all results including retries
    
    while failed_sha256s and retry_count < args.max_retries:
        retry_count += 1
        print(f"\n{'='*100}\n", flush=True)
        print(f"Found {len(failed_sha256s)} failed predictions. Starting retry attempt {retry_count}/{args.max_retries}...\n", flush=True)
        print(f"\n{'='*100}", flush=True)
        print(f"Found {len(failed_sha256s)} failed predictions. Starting retry attempt {retry_count}/{args.max_retries}...", flush=True)
        print(f"Sleeping for {args.retry_delay} seconds before retry...", flush=True)
        time.sleep(args.retry_delay)  # Sleep for retry_delay seconds
        
        # Find the test files corresponding to failed predictions
        retry_files = [f for f in test_files if f.stem in failed_sha256s]
        print(f"Retrying {len(retry_files)} files...", flush=True)
        
        # Reset rate limiter for fresh window
        with mp.Manager() as manager:
            rate_limiter = RateLimiter(manager, tpm_limit=tpm, rpm_limit=rpm)
            print(f"Rate limiter initialized: TPM={tpm}, RPM={rpm}", flush=True)
            
            # Retry predictions
            retry_params = [(f, args, credentials, rate_limiter, pred_dir) for f in retry_files]
            with mp.Pool(args.num_procs) as pool:
                retry_results = pool.map(predict, tqdm(retry_params, total=len(retry_params), desc=f"Retry {retry_count}"))
        
        print(f"Retry {retry_count} complete. Failed: {len(failed_sha256s)}", flush=True)
        # Update results: replace old failed results with new retry results
        predicted_sha256s.extend([r['sha256'] for r in retry_results if r.get("exception") is None])
        all_results.extend(retry_results)
        
        # Check for still-failed predictions
        failed_sha256s = [r['sha256'] for r in retry_results if r.get("exception") is not None]
        
        if failed_sha256s:
            print(f"After retry {retry_count}: {len(failed_sha256s)} predictions still failed", flush=True)
        else:
            print(f"All predictions succeeded after retry {retry_count}!", flush=True)
    
    print("Exiting retry loop...", flush=True)
    
    # Summary
    print(f"\nSummary:", flush=True)
    print(f"  Total predictions attempted: {len(all_results)}", flush=True)
    print(f"  Successful: {len(predicted_sha256s)}", flush=True)
    print(f"  Failed: {len(failed_sha256s)}", flush=True)
    print(f"  Retry attempts: {retry_count}", flush=True)
    print("Predictions done!", flush=True)
    end_time = time.time()
    print(f"Total prediction time: {end_time - start_time:.2f} seconds", flush=True)



if __name__ == "__main__":
    main()