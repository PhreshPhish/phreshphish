import openai
import json
import csv
import re
from bs4 import BeautifulSoup
from openai import OpenAI
import os
import sys
import glob
import myopenaikey
from datetime import datetime
import tiktoken

client = OpenAI(api_key=myopenaikey.mykey)
local_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoint")

MAX_REQUESTS = 10000
MAX_FILE_SIZE_MB = 190
MAX_TOKENS_PER_REQUEST = 3000000
FILE_NAME = "requests_target.jsonl"
CUSTOM_ID_TRACKER_FILE_NAME = "custom_id_tracker.json"
input_file = "gpt-4o-mini-input_phresh.json"
output_file = "gpt-4o-mini-output_phresh.csv"
json_output_file = "output_phresh.json"
raw_response_file = "gpt-4o-mini_phresh.json"
temp_path = '/phishphresh/test/'

system_prompt = """
You are an expert in analyzing URL and multi-lingual HTML to classify webpage as phishing or legitimate page.
Focus only on identifying if the page is phishing and its corresponding risk score.
Given the URL and HTML, perform the following analysis for any Social Engineering techniques often used in phishing attacks:
1.  **PHISHING:** Whether the webpage is a phishing page or a legitimate page. (True if phishing, False if legitimate)
2.  **SCORE:** Indicates phishing risk on a scale of 1 to 10 (inclusive), where 1 is the least likely to have phishing content and 10 is the most likely.
3.  **REASON:** Briefly (one sentence) explain the reasoning behind your determination.
Provide the extracted information in JSON format with the following keys:
- phishing: boolean (whether the site is a phishing site or a legitimate site)
- score: int (indicates phishing risk on a scale of 1 to 10)
- reason: str (one sentence reason)
"""


def save_requests_to_file(requests, selected_date, file_index):
    file_name = f"requests_{selected_date}_{file_index}.jsonl"
    with open(file_name, "w") as file:
        for request in requests:
            file.write(json.dumps(request) + "\n")
    return file_name


def save_custom_id_tracker(custom_id_tracker, selected_date):
    with open(f"custom_id_tracker_{selected_date}.json", "a") as file:
        json.dump(custom_id_tracker, file, indent=4)


def load_input_data(selected_date):
    input_file_name = f"input_{selected_date}.json"
    local_file_path = os.path.join(local_directory, input_file_name)

    if os.path.exists(local_file_path):
        print(f"File {input_file_name} already exists locally. Reading from local directory.")
        with open(local_file_path, 'r') as f:
            data = json.load(f)

    return data


def lengthToken(html):
    return len(re.findall(r'\S+', html))


def countToken(html):
    # Load tokenizer for the model
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    # Count tokens
    token_count = len(encoding.encode(html))
    
    return token_count


def SimplifyHTML(inputHTML):
    soup = BeautifulSoup(inputHTML, "html.parser")

    for tag in soup(["style", "script"]):
        tag.decompose()

    for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.startswith("<!--")):
        comment.extract()

    processedHTML = str(soup)

    if countToken(processedHTML) < 2500:
        return processedHTML

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

    if countToken(processedHTML) < 2500:
        return processedHTML

    while countToken(processedHTML) > 2500:
        tags = soup.find_all(True)
        if not tags:
            break
        midpoint = len(tags) // 2
        tags[midpoint].decompose()
        processedHTML = str(soup)

    return processedHTML


def estimate_tokens(text):
    """Rough estimation: 1 word = ~1.3 tokens"""
    return int(len(text.split()) * 1.3)


def create_requests_files(selected_date):
    # data = load_input_data(selected_date)
    requests = []
    custom_id_tracker = {}
    file_index = 1
    system_tokens = countToken(system_prompt)

    for file_name in os.listdir(temp_path):
        local_file_path = os.path.join(temp_path, file_name)
        with open(local_file_path, 'r') as f:
            data = json.load(f)

        url = data.get("url", "")
        html_content = data.get("html_content", data.get("html", data.get("content", "")))
        groundtruth = data.get("groundtruth", 1)

        if html_content:
            # soup = BeautifulSoup(html_content, 'html.parser')
            # processedHTML = soup.get_text()
            processedHTML = SimplifyHTML(html_content)   
            user_message = f"URL: {url}\nHTML: {processedHTML}"
            user_tokens = countToken(user_message)

            custom_id = f"{selected_date}-{file_index}-{len(requests) + 1}"
            requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ]
                }
            })
            custom_id_tracker[custom_id] = {
                "url": url,
                "groundtruth": groundtruth
            }

        if len(requests) >= MAX_REQUESTS or os.path.getsize(save_requests_to_file(requests, selected_date, file_index)) / (1024 * 1024) > MAX_FILE_SIZE_MB or system_tokens + user_tokens > MAX_TOKENS_PER_REQUEST:
            save_requests_to_file(requests, selected_date, file_index)

            file_index += 1
            requests = []

    if requests:
        save_requests_to_file(requests, selected_date, file_index)

    save_custom_id_tracker(custom_id_tracker, selected_date)


def create_batch(selected_date, file_index):
    batch_input_file = client.files.create(file=open(f"requests_{selected_date}_{file_index}.jsonl", "rb"), purpose="batch")
    batch_input_file_id = batch_input_file.id

    batch_create_response = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "Batch job for classifying phishing or legitimate"
            }
        )

    batch_id = batch_create_response.id
    batch_retrieve_response = client.batches.retrieve(batch_id)
    status = batch_retrieve_response.status
    print(f"Current batch status: {status}")


def show_batch_output(selected_date):
    response = client.batches.list(limit=5)
    completed = False
    batch_data = response.data[0]
    batch_id = batch_data.id
    input_file_id = batch_data.input_file_id
    output_file_id = batch_data.output_file_id
    status = batch_data.status
    error_file_id = batch_data.error_file_id

    print(f"Batch ID: {batch_id}")
    print(f"status:{status}")
    print(f"Input File ID: {input_file_id}")
    print(f"Output File ID: {output_file_id}")

    if status == 'completed':
        file_response = client.files.content(output_file_id).content
        with open(f"response_{selected_date}_{batch_id}.jsonl", "ab") as jsonfile:
            jsonfile.write(file_response)
        print("Batch processing completed and response saved.")

        if error_file_id:
            errors = client.files.content(error_file_id).content
            with open(f"errors_{selected_date}_{batch_id}.jsonl", "ab") as jsonfile:
                jsonfile.write(errors)
        
        completed = True

    elif status in ['failed', 'cancelled', 'expired']:
        if error_file_id:
            errors = client.files.content(error_file_id).content
            with open(f"errors_{selected_date}_{batch_id}.jsonl", "ab") as jsonfile:
                jsonfile.write(errors)

        print(f"Batch processing failed with status: {status}")
        completed = True

    elif status == "in_progress":
        print(f"Current batch status: {status}")
        completed = False

    return completed, batch_id


def process_batch_document(selected_date, batch_id):
    with open(f"custom_id_tracker_{selected_date}.json", "r") as file:
        loaded_tracker = json.load(file)

    raw_responses = []
    extracted_output_data = []
    with open(f"gpt-4o-mini-output_{selected_date}_{batch_id}.csv", "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["url", "groundtruth", "phishing", "score", "reason"])
        total_prompt_tokens = 0
        total_completion_tokens = 0
        price_for_prompt_tokens = 0.075 # per 1M tokens
        price_for_output_tokens = 0.300 # per 1M tokens

        with open(f"response_{selected_date}_{batch_id}.jsonl", 'r') as file:
            for line in file:
                output = json.loads(line.strip())
                custom_id = output.get("custom_id")
                usage = output["response"]["body"]["usage"]
                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)
                gpt_response = output["response"]["body"]["choices"][0]["message"]["content"]
                cleaned_json = gpt_response.replace('```json', '').replace('```', '').strip()
                json_match = re.search(r'\{.*?\}', cleaned_json, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        parsed_json = json.loads(json_str)
                    except json.JSONDecodeError:
                        print("Failed to decode JSON.")
                else:
                    parsed_json = json.loads(cleaned_json)
                phishing = parsed_json.get('phishing', 'N/A')
                score = parsed_json.get('score', 'N/A')
                reason = parsed_json.get('reason', 'N/A')
                if custom_id in loaded_tracker:
                    url = loaded_tracker[custom_id]["url"]
                    groundtruth = loaded_tracker[custom_id]["groundtruth"]

                csv_writer.writerow([url, groundtruth, phishing, score, reason])

                extracted_output_entry = {
                    'url': url,
                    'ground_truth': groundtruth,
                    'phishing': phishing,
                    'score': score,
                    'reason': reason
                }

                extracted_output_data.append(extracted_output_entry)
                raw_responses.append(parsed_json)

    usage = (total_prompt_tokens / 1000000) * price_for_prompt_tokens + (total_completion_tokens / 1000000) * price_for_output_tokens
    print(f"Total cost of running the batch API is {usage} dollars")


def delete_batch_output():
    # running_statuses = ["validating", "in_progress", "finalizing", "cancelling"]
    response = client.batches.list(limit=5)
    for batch_data in response.data:
        batch_id = batch_data.id
        status = batch_data.status
        if status == "completed":
            batch_delete_response = client.batches.delete(batch_id)
            status = batch_delete_response.status
            print(f"Current batch status: {status}")


def cancel_batch_output():
    # running_statuses = ["validating", "in_progress", "finalizing", "cancelling"]
    response = client.batches.list(limit=5)
    for batch_data in response.data:
        batch_id = batch_data.id
        status = batch_data.status
        if status == "in_progress":
            batch_delete_response = client.batches.cancel(batch_id)
            status = batch_delete_response.status
            print(f"Current batch status: {status}")


def main(selected_date_str):
    create_requests_files(selected_date_str)
    create_batch(selected_date_str)
    show_batch_output(selected_date_str)
    process_batch_document(selected_date_str)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)

    try:
        selected_date_str = sys.argv[1]
        option = sys.argv[2]
        
        if len(sys.argv) > 2:
           file_index = sys.argv[3]        

        if option == "create_requests":
            create_requests_files(selected_date_str)

        elif option == "create_batch":
            create_batch(selected_date_str, file_index)

        elif option == "show":
            is_output, batch_id = show_batch_output(selected_date_str)
            if is_output is True:
                print("Now processing the batch output")
                process_batch_document(selected_date_str, batch_id)

        elif option == "delete":
            delete_batch_output()

        elif option == "cancel":
            cancel_batch_output()

        else:
            print("invalid option")

    finally:
        print("To be cleanedup")
