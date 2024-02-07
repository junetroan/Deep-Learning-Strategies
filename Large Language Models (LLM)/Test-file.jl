using OpenAI
key = "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"
model = "gpt-3.5"

start_prompt = "Write a summary of time series prediction."
r = create_chat(key, model, start_prompt; max_tokens=50, stop_sequence=".")
gpt_resp = r.response["choices"][1]["text"]