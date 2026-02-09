from datasets import load_dataset

# Load dataset WITHOUT audio
print("Loading AMI dataset (text only)...")
dataset = load_dataset("edinburghcstr/ami", "ihm")

# Remove audio column immediately
dataset = dataset.remove_columns(['audio'])

print("\n=== Dataset loaded (no audio) ===")
print(f"Train: {len(dataset['train'])} utterances")
print(f"Validation: {len(dataset['validation'])} utterances")
print(f"Test: {len(dataset['test'])} utterances")

# Export to text files by meeting
print("\nExporting text files...")
for split in ['train', 'validation', 'test']:
    for item in dataset[split]:
        meeting_id = item['meeting_id']
        speaker_id = item['speaker_id']
        text = item['text']
        begin_time = item['begin_time']
        
        # Append to meeting file
        with open(f'data/{meeting_id}_transcript.txt', 'a', encoding='utf-8') as f:
            f.write(f"{speaker_id}|{begin_time}|{text}\n")

print("Done! Text files saved in data/ folder")