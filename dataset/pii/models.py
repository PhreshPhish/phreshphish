import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"


EXAMPLES = [
    {
        "role": "user",
        "content": (
            "<url>https://variety.com/t/adam-driver/</url>" "<name>Adam Driver</name>"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Adam Driver is a well-known actor and Variety is a well-known entertainment news website. Almost certainly not a customer.  "
            '<answer>false</answer>.'
        ),
    },
    {
        "role": "user",
        "content": (
            "<url>https://obituaries.goshennews.com/obituary/ryan-ferris-1067496566</url>"
            "<name>Ryan Ferris</name>"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "URL suggests that this is an obituary page, its unlikely that a customer was viewing their own obituary. Probably not a customer. "
            '<answer>false</answer>.'
        ),
    },
    {
        "role": "user",
        "content": (
            "<url>https://lawyers.justia.com/lawyer/ann-marie-mchale-1182248</url>"
            "<name>Ann Marie McHale</name>"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "URL suggests that this is a lawyer profile page and lawyers are public individuals. Probably not a customer. "
            '<answer>false</answer>.'
        ),
    },
    {
        "role": "user",
        "content": (
            "<url>https://www.linkedin.com/in/shawn-johnson-8nasldkn1</url>"
            "<name>Shawn Johnson</name>"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "URL suggests that this is a LinkedIn profile page, this could be from a customer viewing their own profile. "
            '<answer>true</answer>.'
        ),
    },
]

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "Users will provide a URL and a name derived from the URL. "
        "These URLs are sourced from customer traffic logs and we want to determine if the name is likely to be a name belonging to our customer. "
        "Your task is to determine if the name belongs to our customer. "
        'Well-known people, such as actors, politicians, and public figures, are not customers. '
        "If the name is likely to be a customer, respond with <answer>true</answer>. "
        "If the name is not likely to be a customer, respond with <answer>false</answer>. "
        "If you are unsure, respond with <answer>false</answer>. "
        "Be concise."
    ),
}


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        message = [
            {
                "role": "user",
                "content": question,
            },
        ]
        return self.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompt: str) -> str:
        prompt = self.format_prompt(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[0, prompt_len:]
        completion = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        return completion

    def parse_response(self, response: str) -> str:
        """
        Parse the <answer></answer> tag and return a string.
        """
        try:
            return response.split("<answer>")[1].split("</answer>")[0]
        except (IndexError, ValueError):
            return "ERROR"

    def answer(self, question) -> str:
        """
        Answer a question.
        """
        response = self.generate(self.format_prompt(question))
        return self.parse_response(response)


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        message = [SYSTEM_MESSAGE, *EXAMPLES, {"role": "user", "content": question}]

        return self.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )


def load() -> CoTModel:
    return CoTModel()
