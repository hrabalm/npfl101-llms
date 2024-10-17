# see also https://huggingface.co/docs/transformers/llm_tutorial
# and https://huggingface.co/docs/transformers/llm_tutorial_optimization

# you might also consider using other inference libraries, such as vllm or text-generation-inference

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# MODEL = "/storage/brno12-cerit/home/hrabalm/models/npfl101_test_model"
MODEL = "google/gemma-2-2b"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
)

input_text = f"Translate the following Czech sentence to English.\nCzech: Praha je hlavní město Česka.\nEnglish:",
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))

# Optional exercises:
# - Try using pipeline and batching instead https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching
#   - You could also handle the batching by yourself
# - Look into what device_map="auto" does
# - If you need early stopping, look into custom StoppingCriteria, note that handling batching can make it a bit more complex (you need to stop only when all batches are to supposed to stop)
# - Experiment with your tokenizer, how well does it tokenize your language? Do spaces and newlines in the prompt matter?
# - You can also experiment with in-context learning. Try providing the model with several examples and expected output and see how it behaves.
