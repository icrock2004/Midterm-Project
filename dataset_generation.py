from datasets import load_dataset
import json

dataset_name = 'andrewsiah/personalization_promptresponse'
n = 4000
output_file = 'dataset.txt'


def format_record(name: str, prompt: str, output: str) -> str:
    return f"({json.dumps(name)}, {json.dumps(prompt)}, {json.dumps(output)})\n"

def main():
    data = load_dataset(dataset_name)
    dataset = data['train']
 
    # Only take the first N examples
    subset = dataset.select(range(min(n, len(dataset))))
 
    with open(output_file, "w", encoding="utf-8") as out:
        for example in subset:
            prompt = example["prompt"]
            model_1 = example["response_3_model"]
            response_1 = example["response_3"]

            model_2 = example["response_5_model"]
            response_2 = example["response_5"]

            model_3 = example["response_8_model"]
            response_3 = example["response_8"]

            if (model_1 == "anthropic/claude-3-opus" and model_2 == "openai/gpt-4o" and model_3 == "google/gemini-pro-1.5"):
                out.write(format_record(model_1, prompt, response_1))
                out.write(format_record(model_2, prompt, response_2))
                out.write(format_record(model_3, prompt, response_3))
 
    print(f"Wrote {len(subset)} records to {output_file}")
 
 
if __name__ == "__main__":
    main()