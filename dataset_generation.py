from datasets import load_dataset
import json

dataset_name = 'lmsys/chatbot_arena_conversations'
n = 10000
threshold = 950
output_file = 'dataset.txt'


def format_record(model: str, input: str, output: str) -> str:
    return f"({json.dumps(model)}, {json.dumps(input)}, {json.dumps(output)})\n"

def main():
    data = load_dataset(dataset_name)
    dataset = data['train']

    claudeCount = 0
    gptCount = 0
    koalaCount = 0
    # Only take the first N examples
    #subset = dataset.select(range(min(n, len(dataset))))
    
 
    with open(output_file, "w", encoding="utf-8") as out:
        for example in dataset:
            if not example["language"] == "English":
                continue
            
            model = example["model_a"]
            response = example["conversation_a"]
            prompt = response[0]["content"]
            output = response[1]["content"]

            if model == "koala-13b" and koalaCount < threshold:
                out.write(format_record(model, prompt, output))
                koalaCount+=1

            if model == "claude-v1" and claudeCount < threshold:
                out.write(format_record(model, prompt, output))
                claudeCount+=1

            if model == "gpt-4" and gptCount < threshold:
                out.write(format_record(model, prompt, output))
                gptCount+=1
                
 
    print(f"Wrote {koalaCount} koala records to {output_file}\n")
    print(f"Wrote {gptCount} koala records to {output_file}\n")
    print(f"Wrote {claudeCount} koala records to {output_file}\n")

 
 
if __name__ == "__main__":
    main()