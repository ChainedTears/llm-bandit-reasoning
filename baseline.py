import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import secrets
from huggingface_hub import login
import re

login(token="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")

previous_outputs = ""

prompt_two = f"""You are a decision-making agent. Your task is to choose between slot machine 1 or 2.
Based on the history of wins and losses, decide which machine to play next.
Output ONLY the number '1' or the number '2'. Do not include any other words, explanations, or formatting.

Example 1:
# Analysis: Machine 2 has 3 wins/1 loss (75%). Machine 1 has 2 wins/2 losses (50%). Machine 2 is better.
History:
Slot Machine 1 won
Slot Machine 2 won
Slot Machine 1 lost
Slot Machine 2 won
Slot Machine 1 won
Slot Machine 2 lost
Slot Machine 1 lost
Slot Machine 2 won
Your choice (1 or 2): 2

Example 2:
# Analysis: Machine 1 has 2 wins/1 loss (67%). Machine 2 has 1 win/2 losses (33%). Machine 1 is better.
History:
Slot Machine 2 lost
Slot Machine 1 won
Slot Machine 2 won
Slot Machine 1 won
Slot Machine 2 lost
Your choice (1 or 2): 1

Current situation:
History:
{previous_outputs}
Your choice (1 or 2):"""


prompt_three = f"""You are a decision-making agent. Your task is to choose the slot machine most likely to win.
Based on the history of wins and losses, decide which machine to play next.
Output ONLY the number 1 or 2 or 3. Do not include any other words or formatting.

Example 1:
# Analysis: Machine 3 has 3 wins/1 loss (75%). Machine 1 has 2 wins/1 loss (67%). Machine 2 has 0 wins/2 losses (0%). Machine 3 is the best.
History:
Slot Machine 1 won
Slot Machine 2 lost
Slot Machine 3 won
Slot Machine 1 won
Slot Machine 3 won
Slot Machine 2 lost
Slot Machine 3 lost
Slot Machine 3 won
Your choice (1, 2, or 3): 3

Example 2:
# Analysis: Machine 1 has 2 wins/0 losses (100%). Machine 3 has 2 wins/1 loss (67%). Machine 2 has 1 win/2 losses (33%). Machine 1 is the best.
History:
Slot Machine 2 won
Slot Machine 1 won
Slot Machine 3 won
Slot Machine 2 lost
Slot Machine 3 lost
Slot Machine 1 won
Slot Machine 2 lost
Slot Machine 3 won
Your choice (1, 2, or 3): 1

Current situation:
History:
{previous_outputs}
Your choice (1, 2, or 3):"""


prompt_four = f"""You are a decision-making agent. Your task is to choose the slot machine most likely to win.
Based on the history of wins and losses, decide which machine to play next.
Output ONLY the number 1 or 2 or 3 or 4. Do not include any other words or formatting.

Example 1:
# Analysis: Machine 4 has 2 wins/0 losses (100%). Machine 1 has 1 win/1 loss (50%). Machine 2 has 1 win/1 loss (50%). Machine 3 has 0 wins/2 losses (0%). Machine 4 is the best.
History:
Slot Machine 1 won
Slot Machine 2 lost
Slot Machine 3 lost
Slot Machine 4 won
Slot Machine 1 lost
Slot Machine 2 won
Slot Machine 3 lost
Slot Machine 4 won
Your choice (1, 2, 3, or 4): 4

Example 2:
# Analysis: Machine 1 has 3 wins/0 losses (100%). Machine 2 has 2 wins/1 loss (67%). Machine 3 has 1 win/1 loss (50%). Machine 4 has 0 wins/2 losses (0%). Machine 1 is the best.
History:
Slot Machine 2 won
Slot Machine 3 won
Slot Machine 1 won
Slot Machine 4 lost
Slot Machine 2 won
Slot Machine 1 won
Slot Machine 3 lost
Slot Machine 4 lost
Slot Machine 1 won
Slot Machine 2 lost
Your choice (1, 2, 3, or 4): 1

Current situation:
History:
{previous_outputs}
Your choice (1, 2, 3, or 4):"""


prompt_five = f"""You are a decision-making agent. Your task is to choose the slot machine most likely to win.
Based on the history of wins and losses, decide which machine to play next.
Output ONLY the number 1 or 2 or 3 or 4 or 5. Do not include any other words or formatting.

Example 1:
# Analysis: Machine 2 has 3 wins/0 losses (100%). Machine 5 has 2 wins/1 loss (67%). Others are worse. Machine 2 is the best.
History:
Slot Machine 5 won
Slot Machine 1 lost
Slot Machine 2 won
Slot Machine 4 lost
Slot Machine 5 won
Slot Machine 2 won
Slot Machine 3 lost
Slot Machine 5 lost
Slot Machine 2 won
Your choice (1, 2, 3, 4, or 5): 2

Example 2:
# Analysis: Machine 5 has 2 wins/0 losses (100%). Machine 3 has 2 wins/1 loss (67%). Others are worse. Machine 5 is the best.
History:
Slot Machine 3 won
Slot Machine 1 lost
Slot Machine 2 won
Slot Machine 4 lost
Slot Machine 5 won
Slot Machine 2 lost
Slot Machine 3 won
Slot Machine 1 lost
Slot Machine 5 won
Slot Machine 3 lost
Your choice (1, 2, 3, 4, or 5): 5

Current situation:
History:
{previous_outputs}
Your choice (1, 2, 3, 4, or 5):"""

prompt_dict = {
    '1': prompt_two,
    '2': prompt_three,
    '3': prompt_four,
    '4': prompt_five,
}

# Specify model ID 
# model_id = "meta-llama/Llama-3.2-1B"
model_dict = {
    '1': "Qwen/Qwen3-4B",
    '2': "Qwen/Qwen3-8B",
    '3': "meta-llama/Llama-3.1-8B",
    '4': "deepseek-ai/DeepSeek-R1",
    '5': "microsoft/phi-2",
    '6': "google/gemma-3-12b-it",
}

start_match_pattern_dict = {
    prompt_two: r'^\s*([12])\b',
    prompt_three: r'^\s*([1-3])\b',
    prompt_four: r'^\s*([1-4])\b',
    prompt_five: r'^\s*([1-5])\b'
}

fallback_match_pattern_dict = {
    prompt_two: r'\b([12])\b',
    prompt_three: r'\b([1-3])\b',
    prompt_four: r'\b([1-4])\b',
    prompt_five: r'\b([1-5])\b'
}

valid_choices_dict = {
    prompt_two: [1, 2],
    prompt_three: [1, 2, 3],
    prompt_four: [1, 2, 3, 4],
    prompt_five: [1, 2, 3, 4, 5]
}

# receive = input("Please select the model (using a number from 1-7): \n (1) Qwen 4B \n (2) Qwen 8B \n (3) Llama 8B \n (4) Mistral 7B \n (5) Phi 2 \n (6) Gemma 3 12B \n Select here: ")
# while receive not in model_dict:
#     receive = input("Please select the model: \n (1) Qwen 4B \n (2) Qwen 8B \n (3) Llama 8B \n (4) Mistral 7B \n (5) Phi 2 \n (6) Gemma 3 12B \n Select here: ")
# model_id = model_dict[receive]

# receive = input("Please select the option amount (using a number from 1-4): \n (1) Two \n (2) Three \n (3) Four \n (4) Five \n Select here: ")
# while receive not in prompt_dict:
#     receive = input("Please select the option amount (using a number from 1-4): \n (1) Two \n (2) Three \n (3) Four \n (4) Five \n Select here: ")
# prompt_type = prompt_dict[receive]

for model in list(model_dict.values()):
    for prompt in list(prompt_dict.values()):
        print(f"Testing {model} with {prompt}")
        model_id = model
        prompt_type = prompt

        best_machine_dict = {
            prompt_two: 2,
            prompt_three: 3,
            prompt_four: 1,
            prompt_five: 3
        }

        best_machine = best_machine_dict[prompt_type]
        start_match_pattern = start_match_pattern_dict[prompt_type]
        fallback_match_pattern = fallback_match_pattern_dict[prompt_type]
        valid_choices = valid_choices_dict[prompt_type]
        # Setup device (MPS for Mac, CUDA, fallback to CPU)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=".")
            print("Tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            tokenizer = None

        # Load model
        model = None
        if tokenizer:
            try:
                # Determine torch_dtype based on device
                if device.type == "mps":
                    dtype = torch.float16
                elif device.type == "cuda":
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                        dtype = torch.bfloat16
                    else:
                        dtype = torch.float16
                else: # CPU
                    dtype = torch.float32
                
                print(f"Loading model with dtype: {dtype}")

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    cache_dir=".",
                    # trust_remote_code=True,
                ).to(device)
                # model.eval()  # Set model to evaluation mode
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                model = None

        # Set pad_token_id to eos_token_id if not already set
        # This must be done AFTER the tokenizer is loaded.
        if tokenizer and tokenizer.pad_token_id is None:
            print("Tokenizer does not have a pad_token_id. Setting it to eos_token_id.")
            tokenizer.pad_token_id = tokenizer.eos_token_id
            # For Llama models, padding side is often recommended to be 'left' for batched inference.
            # tokenizer.padding_side = "left" # Keep an eye on this if you do batching

        # Generate response function

        # Add a stop sequence

        def get_response(prompt_text):
            if model is None or tokenizer is None:
                return "Model or tokenizer not loaded."

            # For dedicated chat models, using their chat template is often more robust.
            # Example for Qwen1.5-Chat:
            # messages = [
            #     {"role": "system", "content": "You are a decision-making agent. Your sole task is to choose between slot machine 1 or 2. Respond with ONLY the number '1' or '2'."},
            #     {"role": "user", "content": prompt_text} # prompt_text here would be the history part for the user message
            # ]
            # formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            # For this example, we'll stick to direct prompt construction as in the main loop.

            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[1]

            start_time = time.time()
            # print("Generating response...") # Moved to main loop
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask, # Pass attention_mask
                    max_new_tokens=3,      # CRITICAL: Keep low for single digit output
                    do_sample=True,       # Greedy decoding; set to True with low temp if output is too repetitive
                    temperature=0.7,     # Use if do_sample=True
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            # print(f"Generation took {time.time() - start_time:.2f} seconds.")

            newly_generated_tokens = outputs[0, input_length:]
            generated_text = tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip()
            # The ### regex removal might not be needed if the prompt is clean
            # generated_text = re.sub(r'###.*?###.*?###', '', generated_text, flags=re.DOTALL)
            return generated_text


        # def get_response(prompt):
        #     if model is None or tokenizer is None:
        #         return "Model or tokenizer not loaded."

        #     inputs = tokenizer(prompt, return_tensors="pt").to(device)
        #     # log timestamp
        #     start_time = time.time()
        #     print("Generating response...")
        #     with torch.no_grad():
        #         outputs = model.generate(
        #         **inputs,
        #         max_new_tokens=100,
        #         do_sample=True,
        #         temperature=0.1,
        #         top_p=1.0,
        #         pad_token_id=tokenizer.pad_token_id,  # Use the pad_token_id we set earlier
        #         eos_token_id=tokenizer.eos_token_id
        #     ) 
        #     # Only return *new* generated tokens
        #     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #     print("Generation took ", time.time() - start_time, "seconds.")
        #     generated_text = generated_text.split("<|assistant|>")[-1].strip()
        #     generated_text = re.sub(r'###.*?###.*?###', '', generated_text, flags=re.DOTALL)
        #     return generated_text


        # def bandit_simulation(choice):
        #     random_number = secrets.randbelow(100)
        #     if choice == 1:
        #         if random_number < 30:
        #             return "won"
        #         else:
        #             return "lost"
        #     if choice == 2:
        #         if random_number < 65:
        #             return "won"
        #         else: 
        #             return "lost"


        def bandit_simulation(choice):
            random_number = secrets.randbelow(100)
            if prompt_type == prompt_two:
                if choice == 1: # 30% win rate
                    return "won" if random_number < 30 else "lost"
                if choice == 2: # 65% win rate (objectively better)
                    return "won" if random_number < 65 else "lost"
            elif prompt_type == prompt_three:
                if choice == 1:
                    return "won" if random_number < 40 else "lost"
                if choice == 2:
                    return "won" if random_number < 30 else "lost"
                if choice == 3: # (objectively better)
                    return "won" if random_number < 70 else "lost" 
            elif prompt_type == prompt_four:
                if choice == 1: # (objectively better)
                    return "won" if random_number < 80 else "lost" 
                if choice == 2:
                    return "won" if random_number < 60 else "lost"
                if choice == 3:
                    return "won" if random_number < 35 else "lost"
                if choice == 4:
                    return "won" if random_number < 25 else "lost" 
            elif prompt_type == prompt_five:
                if choice == 1:
                    return "won" if random_number < 20 else "lost"
                if choice == 2:
                    return "won" if random_number < 75 else "lost"
                if choice == 3: # (objectively better)
                    return "won" if random_number < 35 else "lost"
                if choice == 4:
                    return "won" if random_number < 25 else "lost"
                if choice == 5: 
                    return "won" if random_number < 55 else "lost" 
            print(f"Error in bandit_simulation with choice: {choice}")
            return "error"


        global_history = []
        correct_counter = 0

        def main():
            cumulative_reward = 0
            global previous_outputs
            correct_ai_choices, total_ai_decisions, previous_ai_choice = 0, 0, 1 # Metrics for AI

            max_iterations = 25 # Number of decisions the AI will make
            iteration_results = [] 
            for i in range(max_iterations):
                iteration_num = i + 1
                # print(f"------------- Iteration {iteration_num} -------------")

                # Construct the prompt: Strong instructions + Few-shot examples
                prompt = prompt_type.format(previous_outputs=previous_outputs)

                ai_response_raw = get_response(prompt)
                # print(f"Raw AI Response: {ai_response_raw}")

                ai_choice = None
                # Stricter parsing: expect '1' or '2' at the beginning of the response
                match = re.match(start_match_pattern, ai_response_raw)
                if match:
                    try:
                        ai_choice = int(match.group(1))
                    except ValueError:
                        print(f"AI response parsing error (ValueError) from '{ai_response_raw}'.")
                else: # Fallback if no direct 1 or 2 found at the start
                    match_fallback = re.search(fallback_match_pattern, ai_response_raw) # Look for 1 or 2 anywhere
                    if match_fallback:
                        try:
                            ai_choice = int(match_fallback.group(1))
                            print(f"Used fallback regex to find choice: {ai_choice}")
                        except ValueError:
                            print(f"AI response parsing error (ValueError) on fallback from '{ai_response_raw}'.")


                if ai_choice not in valid_choices:
                    print(f"AI did not output a clear number in the list: {valid_choices} Asking again...")
                    while ai_choice not in valid_choices:
                        ai_response_raw = get_response(prompt)
                        match = re.match(start_match_pattern, ai_response_raw)
                        if match:
                            ai_choice = int(match.group(1))



                # print(f"AI chose: Machine {ai_choice}")

                total_ai_decisions += 1
                # Machine 2 is objectively better (65% win rate).
                # Count as "correct" if AI picks machine 2.
                if ai_choice == best_machine:
                    correct_ai_choices += 1
                result = bandit_simulation(ai_choice)
                if result == "won":
                    cumulative_reward += 1
                else:
                    cumulative_reward -= 1
                current_choice_str = f"Slot Machine {ai_choice} {result}\n"
                previous_outputs += current_choice_str # Add current result to history for next turn
                previous_ai_choice = ai_choice # Update previous AI choice
                global correct_counter
                correct_counter += 1 if result == "won" else 0

                # print(f"Outcome: Machine {ai_choice} {result}.")
                current_ratio = correct_ai_choices / total_ai_decisions if total_ai_decisions > 0 else 0
                # print(f"AI 'Correct Choice' (picked Machine {best_machine}) Ratio: {correct_ai_choices}/{total_ai_decisions} = {current_ratio:.2f}")
                print(f"Ratio: {correct_ai_choices}/{total_ai_decisions} = {current_ratio:.2f}")
            
            final_ratio = correct_ai_choices / total_ai_decisions if total_ai_decisions > 0 else 0
            global_history.append([final_ratio, cumulative_reward])
            print("--------------------------------")
            print(f"Final ratio: {final_ratio:.2f}")
                    # For brevity, don't print full history each time
                    # print(f"Updated History (last 5 lines):\n{''.join(previous_outputs.splitlines(True)[-5:])}")

        if __name__ == "__main__":
            for i in range(500):
                previous_outputs = ""
                print(f"------------- Test {i+1} -------------")
                main()
            # Writes output into result.txt; in case connection closes for runpod
            for key, value in prompt_dict.items():
                if value == prompt_type:
                    prompt_key = key

            with open(f"{model_id}-{prompt_key}.txt", "w") as f:
                if isinstance(global_history, list):
                    for item in global_history:
                        f.write(str(item) + ", ")
                else:
                    f.write(str(global_history) + "\n")
                    
                average_ratio = sum(global_history) / len(global_history)
                print(f"Average ratio: {average_ratio:.2f}\n")
                f.write(f"Average ratio: {average_ratio:.2f}\n")
                raw_accuracy = correct_counter / 500
                print(f"Raw accuracy: {raw_accuracy}\n")
                f.write(f"Raw accuracy: {raw_accuracy}\n")
            print("Results have been written to result.txt")
