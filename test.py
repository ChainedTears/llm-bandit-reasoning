# test.py (Main Experiment Script - CORRECTED MODEL LIST)
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import secrets
# from huggingface_hub import login # Ensure you are logged in via CLI or uncomment and use your token
import re
import numpy as np
import copy # For deep copying scenario configurations
import json # For saving results
import os # For creating directories

# --- Hugging Face Login (ensure you're logged in, or use a token if needed) ---
# login(token="YOUR_HF_TOKEN_HERE") 

# --- Scenario Definitions ---
ALL_SCENARIOS = {
    "SlotMachine": {
        "scenario_name": "Classic Slot Machine Challenge",
        "task_description": "Your task is to choose between playing SLOT1 or SLOT2.",
        "option_intro": "Each slot machine can result in a 'WIN' or a 'LOSS'. Their win rates are unknown to you initially.",
        "options": ["SLOT1", "SLOT2"],
        "outcome_names": ["WIN", "LOSS"],
        "outcome_scores": {"WIN": 1, "LOSS": 0},
        "initial_probabilities": {
            "SLOT1": {"WIN": 0.30, "LOSS": 0.70},
            "SLOT2": {"WIN": 0.65, "LOSS": 0.35}
        },
        "examples_for_prompt": """Example 1:
History:
Turn 1: Chose SLOT1, Result: LOSS
Turn 2: Chose SLOT2, Result: WIN
Turn 3: Chose SLOT2, Result: WIN
Turn 4: Chose SLOT1, Result: LOSS
Your choice (SLOT1, SLOT2): SLOT2

Example 2:
History:
Turn 1: Chose SLOT1, Result: WIN
Turn 2: Chose SLOT1, Result: WIN
Turn 3: Chose SLOT2, Result: LOSS
Turn 4: Chose SLOT1, Result: LOSS
Your choice (SLOT1, SLOT2): SLOT1"""
    },
    "StockMarket": {
        "scenario_name": "Stock Market Speculator",
        "task_description": "Your task is to choose which stock (STKA, STKB, or STKC) to invest in for the current quarter.",
        "option_intro": "All stocks can result in 'High Gain', 'Modest Gain', 'Break Even', 'Modest Loss', or 'Significant Loss'.\nStock STKA (AlphaGrowth Tech) is volatile: it has a higher chance than others of 'High Gain' but also a considerable chance of 'Modest Loss' or 'Significant Loss'. Stock STKB (BetaSustain Energy) often provides 'Modest Gain', with some chance of 'High Gain' and a low chance of any 'Loss'. Stock STKC (GammaStable Goods) most frequently results in 'Modest Gain' or 'Break Even', very rarely any 'Loss'.",
        "options": ["STKA", "STKB", "STKC"],
        "outcome_names": ["High Gain", "Modest Gain", "Break Even", "Modest Loss", "Significant Loss"],
        "outcome_scores": {"High Gain": 20, "Modest Gain": 10, "Break Even": 0, "Modest Loss": -5, "Significant Loss": -10},
        "initial_probabilities": {
            "STKA": {"High Gain": 0.15, "Modest Gain": 0.20, "Break Even": 0.15, "Modest Loss": 0.30, "Significant Loss": 0.20},
            "STKB": {"High Gain": 0.25, "Modest Gain": 0.40, "Break Even": 0.20, "Modest Loss": 0.10, "Significant Loss": 0.05},
            "STKC": {"High Gain": 0.05, "Modest Gain": 0.25, "Break Even": 0.50, "Modest Loss": 0.15, "Significant Loss": 0.05},
        },
        "examples_for_prompt": """Example 1:
History:
Q1: Invested STKC, Result: Modest Gain
Q2: Invested STKA, Result: Modest Loss
Q3: Invested STKB, Result: Modest Gain
Your choice (STKA, STKB, STKC): STKB

Example 2:
History:
Q1: Invested STKA, Result: High Gain
Q2: Invested STKA, Result: Break Even
Q3: Invested STKC, Result: Modest Gain
Your choice (STKA, STKB, STKC): STKA"""
    },
    "CropYield": {
        "scenario_name": "Adaptive Crop Selection Strategist",
        "task_description": "Your task is to choose which crop (CROPX, CROPY, or CROPZ) to recommend planting for the upcoming season.",
        "option_intro": "All crops can result in 'Bumper Yield', 'Good Yield', 'Average Yield', 'Poor Yield', or 'Crop Failure'.\nCROPX (MaizeMax) has a good chance of 'Bumper Yield' or 'Good Yield' with ideal rain, but is prone to 'Poor Yield' or 'Crop Failure' in dry spells. CROPY (BeanGuard) more reliably gives 'Average Yield' or 'Good Yield', rarely 'Bumper Yield' but also rarely 'Crop Failure'. CROPZ (HerbElite) is risky; it can achieve 'Bumper Yield' (in terms of profit value) with high market demand, but often results in 'Average Yield' or 'Poor Yield' if demand is low or pests appear.",
        "options": ["CROPX", "CROPY", "CROPZ"],
        "outcome_names": ["Bumper Yield", "Good Yield", "Average Yield", "Poor Yield", "Crop Failure"],
        "outcome_scores": {"Bumper Yield": 30, "Good Yield": 20, "Average Yield": 10, "Poor Yield": -5, "Crop Failure": -15},
        "initial_probabilities": {
            "CROPX": {"Bumper Yield": 0.25, "Good Yield": 0.35, "Average Yield": 0.20, "Poor Yield": 0.15, "Crop Failure": 0.05},
            "CROPY": {"Bumper Yield": 0.05, "Good Yield": 0.35, "Average Yield": 0.45, "Poor Yield": 0.10, "Crop Failure": 0.05},
            "CROPZ": {"Bumper Yield": 0.20, "Good Yield": 0.20, "Average Yield": 0.30, "Poor Yield": 0.20, "Crop Failure": 0.10},
        },
        "examples_for_prompt": """Example 1:
History:
Season 1: Recommended CROPX (Conditions: Average Rainfall, Stable Market), Result: Good Yield
Season 2: Recommended CROPX (Conditions: Dry Spell, Stable Market), Result: Poor Yield
Season 3: Recommended CROPY (Conditions: Dry Spell, Stable Market), Result: Average Yield
Your choice (CROPX, CROPY, CROPZ): CROPY

Example 2:
History:
Season 1: Recommended CROPZ (Conditions: Average Rainfall, Low Herb Market), Result: Average Yield
Season 2: Recommended CROPY (Conditions: Average Rainfall, Stable Market), Result: Good Yield
Season 3: Recommended CROPX (Conditions: Good Rainfall, Stable Market), Result: Bumper Yield
Your choice (CROPX, CROPY, CROPZ): CROPX"""
    },
    "PrizeDoor": {
        "scenario_name": "The Evolving Prize Door Dilemma",
        "task_description": "Your task is to choose one of three doors (DOOR1, DOOR2, or DOOR3) to win the best prize.",
        "option_intro": "All doors can lead to 'Grand Prize', 'Good Prize', 'Modest Prize', 'Small Prize', or 'No Prize'.\nThe host says DOOR1 'often hides a 'Good Prize,' sometimes even the 'Grand Prize' but can also have 'Small Prize' or 'No Prize'.' DOOR2 is a 'real gamble; it has a shot at the 'Grand Prize' but more frequently a 'Small Prize' or 'No Prize'.' DOOR3 is a 'safe bet, usually a 'Modest Prize' or 'Small Prize,' very rarely 'No Prize' and almost never the 'Grand Prize.''",
        "options": ["DOOR1", "DOOR2", "DOOR3"],
        "outcome_names": ["Grand Prize", "Good Prize", "Modest Prize", "Small Prize", "No Prize"],
        "outcome_scores": {"Grand Prize": 100, "Good Prize": 50, "Modest Prize": 20, "Small Prize": 5, "No Prize": 0},
        "initial_probabilities": {
            "DOOR1": {"Grand Prize": 0.15, "Good Prize": 0.35, "Modest Prize": 0.20, "Small Prize": 0.20, "No Prize": 0.10},
            "DOOR2": {"Grand Prize": 0.20, "Good Prize": 0.10, "Modest Prize": 0.10, "Small Prize": 0.30, "No Prize": 0.30},
            "DOOR3": {"Grand Prize": 0.01, "Good Prize": 0.09, "Modest Prize": 0.50, "Small Prize": 0.30, "No Prize": 0.10},
        },
        "examples_for_prompt": """Example 1:
History:
Round 1: Chose DOOR3, Result: Modest Prize
Round 2: Chose DOOR1, Result: Good Prize
Round 3: Chose DOOR2, Result: Small Prize
Your choice (DOOR1, DOOR2, DOOR3): DOOR1

Example 2:
History:
Round 1: Chose DOOR2, Result: Grand Prize
Round 2: Chose DOOR2, Result: No Prize
Round 3: Chose DOOR3, Result: Small Prize
Your choice (DOOR1, DOOR2, DOOR3): DOOR3"""
    },
    "SmugglingRoutes": {
        "scenario_name": "Shifting Sands Smuggling Routes",
        "task_description": "Your task is to choose the best route (NORTH, CENTRAL, or SOUTH) for your next trip.",
        "option_intro": "All routes can lead to 'Clear Success', 'Delayed Success', 'Minor Encounter (Success)', 'Patrol Confrontation (Failure)', or 'Route Compromised (Major Failure)'.\nIntel suggests NORTH Route is 'often a 'Clear Success,'' with a low chance of 'Minor Encounter'. CENTRAL Route is 'unpredictable; it can be a 'Clear Success' but has a higher risk of 'Patrol Confrontation' or even 'Route Compromised'. SOUTH Route often results in 'Minor Encounter (Success)' or 'Delayed Success' due to light patrols, rarely 'Clear Success' but also rarely 'Patrol Confrontation'.",
        "options": ["NORTH", "CENTRAL", "SOUTH"],
        "outcome_names": ["Clear Success", "Delayed Success", "Minor Encounter (Success)", "Patrol Confrontation (Failure)", "Route Compromised (Major Failure)"],
        "outcome_scores": {"Clear Success": 50, "Delayed Success": 30, "Minor Encounter (Success)": 40, "Patrol Confrontation (Failure)": -20, "Route Compromised (Major Failure)": -50},
        "initial_probabilities": {
            "NORTH": {"Clear Success": 0.70, "Delayed Success": 0.10, "Minor Encounter (Success)": 0.15, "Patrol Confrontation (Failure)": 0.04, "Route Compromised (Major Failure)": 0.01},
            "CENTRAL": {"Clear Success": 0.40, "Delayed Success": 0.10, "Minor Encounter (Success)": 0.10, "Patrol Confrontation (Failure)": 0.25, "Route Compromised (Major Failure)": 0.15},
            "SOUTH": {"Clear Success": 0.10, "Delayed Success": 0.30, "Minor Encounter (Success)": 0.45, "Patrol Confrontation (Failure)": 0.10, "Route Compromised (Major Failure)": 0.05},
        },
        "examples_for_prompt": """Example 1:
History:
Trip 1: Took NORTH route, Result: Clear Success
Trip 2: Took CENTRAL route, Result: Patrol Confrontation (Failure)
Trip 3: Took SOUTH route, Result: Minor Encounter (Success)
Your choice (NORTH, CENTRAL, SOUTH): NORTH

Example 2:
History:
Trip 1: Took CENTRAL route, Result: Clear Success
Trip 2: Took CENTRAL route, Result: Clear Success
Trip 3: Took NORTH route, Result: Minor Encounter (Success)
Your choice (NORTH, CENTRAL, SOUTH): CENTRAL"""
    },
    "DietaryAdvisor": {
        "scenario_name": "Adaptive Dietary Advisor",
        "task_description": "Your task is to choose between recommending DIETP, DIETM, or DIETV to the current client.",
        "option_intro": "All diets can lead to user reports of 'Excellent Results & Adherence', 'Good Results & Adherence', 'Modest Results & Adherence', 'No Benefit & Adherence', or 'Found Difficult (Stopped)'.\nDIETP (Paleo) often leads to 'Excellent Results & Adherence' quickly for some, but a notable portion report 'Found Difficult (Stopped)'. DIETM (Mediterranean) typically shows 'Good Results & Adherence' or 'Modest Results & Adherence' steadily, with few 'Found Difficult (Stopped)'. DIETV (Vegetarian) users often report 'Good Results & Adherence' or 'Modest Results & Adherence)', but some report 'No Benefit & Adherence' or 'Found Difficult (Stopped)' if not planned well.",
        "options": ["DIETP", "DIETM", "DIETV"],
        "outcome_names": ["Excellent Results & Adherence", "Good Results & Adherence", "Modest Results & Adherence", "No Benefit & Adherence", "Found Difficult (Stopped)"],
        "outcome_scores": {"Excellent Results & Adherence": 25, "Good Results & Adherence": 15, "Modest Results & Adherence": 5, "No Benefit & Adherence": 0, "Found Difficult (Stopped)": -10},
        "initial_probabilities": {
            "DIETP": {"Excellent Results & Adherence": 0.30, "Good Results & Adherence": 0.20, "Modest Results & Adherence": 0.10, "No Benefit & Adherence": 0.10, "Found Difficult (Stopped)": 0.30},
            "DIETM": {"Excellent Results & Adherence": 0.15, "Good Results & Adherence": 0.40, "Modest Results & Adherence": 0.30, "No Benefit & Adherence": 0.10, "Found Difficult (Stopped)": 0.05},
            "DIETV": {"Excellent Results & Adherence": 0.10, "Good Results & Adherence": 0.25, "Modest Results & Adherence": 0.30, "No Benefit & Adherence": 0.15, "Found Difficult (Stopped)": 0.20},
        },
        "examples_for_prompt": """Example 1:
History:
Client A: Recommended DIETM, Reported: Good Results & Adherence
Client B: Recommended DIETP, Reported: Excellent Results & Adherence
Client C: Recommended DIETV, Reported: Modest Results & Adherence
Your choice (DIETP, DIETM, DIETV): DIETP

Example 2:
History:
Client D: Recommended DIETP, Reported: Difficult to Adhere (Stopped)
Client E: Recommended DIETM, Reported: Good Results & Adherence
Client F: Recommended DIETV, Reported: No Benefit & Adherence
Your choice (DIETP, DIETM, DIETV): DIETM"""
    }
}

# --- Model Setup (Using YOUR EXACT original list) ---
model_dict = {
    '1': "Qwen/Qwen3-4B",
    '2': "Qwen/Qwen3-8B",
    '3': "meta-llama/Llama-3.1-8B", # Note: Llama-3-8B-Instruct is usually preferred for tasks like this. This base model might require more careful prompting.
    '4': "deepseek-ai/DeepSeek-R1", # NOTE: May require trust_remote_code=True during loading.
    '5': "microsoft/phi-2",         # NOTE: May require trust_remote_code=True during loading.
    '6': "google/gemma-3-12b-it", # WARNING: "gemma-3-12b-it" is not a recognized standard Hugging Face model ID.
                                  # Valid Gemma 2 examples: "google/gemma-2-9b-it", "google/gemma-7b-it".
                                  # This will likely fail to load unless it's a private model ID you have access to.
    '7': "openai/whisper-large-v3"  # WARNING: This is a SPEECH-TO-TEXT model, NOT suitable for this task. It will likely error or give nonsensical results.
}

print("Please select the LLM model (using a number from your original list):")
print(" (1) Qwen/Qwen3-4B")
print(" (2) Qwen/Qwen3-8B")
print(" (3) meta-llama/Llama-3.1-8B")
print(" (4) deepseek-ai/DeepSeek-R1 (May need trust_remote_code=True)")
print(" (5) microsoft/phi-2 (May need trust_remote_code=True)")
print(" (6) google/gemma-3-12b-it (WARNING: Potentially invalid ID or private model!)")
print(" (7) openai/whisper-large-v3 (WARNING: Speech-to-text model! Unsuitable!)")

receive_model_key = input("Select here: ")
while receive_model_key not in model_dict:
    print("Invalid selection. Please choose from the list.")
    receive_model_key = input("Select here: ")
model_id = model_dict[receive_model_key]
print(f"Selected model: {model_id}")

# Setup device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load tokenizer
tokenizer = None
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=".")
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer for {model_id}: {e}")
    if "whisper" in model_id.lower():
        print("Hint: Whisper models are for speech processing and have different tokenizers/usage patterns than text generation models.")
    if "gemma-3" in model_id.lower():
        print(f"Hint: '{model_id}' might be an incorrect ID or a private model. Standard Gemma models are like 'google/gemma-2-9b-it'.")


# Load model
model = None
if tokenizer:
    if "openai/whisper" in model_id: 
        print(f"ERROR: {model_id} is a Whisper (speech-to-text) model and cannot be loaded with AutoModelForCausalLM for this text generation task. Please choose a generative text model for subsequent runs.")
    elif "gemma-3" in model_id.lower() and not os.path.exists(os.path.join(".", model_id.split("/")[-1])): # Basic check if it's not a local path for a private model
        print(f"WARNING: Model ID '{model_id}' appears to be non-standard and might fail to load from Hugging Face Hub. If it's a private model, ensure it's accessible.")
        # We can still attempt to load it, but it's likely to fail if not a valid public HF ID.
        # To be safe, we could skip loading here, but per user request to use their list, we'll try.

    # Common loading logic for generative models
    if model is None and "openai/whisper" not in model_id : # Ensure we don't try if it's Whisper or already failed
        try:
            if device.type == "mps": dtype = torch.float16
            elif device.type == "cuda": dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            else: dtype = torch.float32
            
            print(f"Loading model {model_id} with dtype: {dtype}")
            model_load_args = {"torch_dtype": dtype, "cache_dir": "."}
            
            if model_id in ["microsoft/phi-2", "deepseek-ai/DeepSeek-R1"] or "deepseek" in model_id: # Added general deepseek check
                 model_load_args["trust_remote_code"] = True
                 print("Note: trust_remote_code=True used for this model.")

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_load_args
            ).to(device)
            model.eval() 
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            if "trust_remote_code" not in model_load_args and (model_id in ["microsoft/phi-2", "deepseek-ai/DeepSeek-R1"] or "deepseek" in model_id):
                print("Hint: This model might require 'trust_remote_code=True' during loading.")
            if "gemma-3" in model_id.lower():
                 print(f"Hint: Loading for '{model_id}' failed. This might be due to an incorrect ID. Standard Gemma models are like 'google/gemma-2-9b-it'.")


if tokenizer and tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
        print("Tokenizer does not have a pad_token_id. Setting it to eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        print("Warning: Tokenizer has no pad_token_id and no eos_token_id. Adding a new pad_token '[PAD]'.")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if model: 
             model.resize_token_embeddings(len(tokenizer))


# --- LLM Interaction Functions ---
def get_llm_response(prompt_text):
    if model is None or tokenizer is None:
        return "ERROR_MODEL_NOT_LOADED"
    if tokenizer.pad_token_id is None: 
        return "ERROR_PAD_TOKEN_NONE"
    if "openai/whisper" in model_id: # Double check here
        return "ERROR_WRONG_MODEL_TYPE_FOR_TASK"

    # Truncation Strategy
    # Max model length (e.g., 4096, 8192). Reserve space for generation (e.g., 60 tokens).
    effective_max_len = tokenizer.model_max_length - 60 if tokenizer.model_max_length else 2048 - 60 # Fallback if no model_max_length

    # Tokenize once to check length
    inputs_dict = tokenizer(prompt_text, truncation=False, return_tensors=None) # No tensors yet
    token_ids = inputs_dict['input_ids']

    if len(token_ids) > effective_max_len:
        # print(f"Warning: Prompt too long ({len(token_ids)} tokens vs max {effective_max_len}). Truncating.")
        # Preserve initial instructions & examples, and most recent history.
        # This is heuristic. A better way involves tokenizing line by line or by sections.
        lines = prompt_text.split('\n')
        header_end_line_idx = 0
        for idx, line in enumerate(lines):
            if "Current situation:" in line: # Assumes "Current situation:" is after examples
                header_end_line_idx = idx
                break
        if header_end_line_idx == 0 and len(lines) > 25: # Fallback if "Current situation:" not found early
            header_end_line_idx = 15 # Keep more of the top if no clear marker

        # How many lines of recent history to keep (rough estimate)
        # Each history line is ~10-15 tokens. Target ~1000-1500 tokens for recent history.
        # This part is very tricky to do perfectly without knowing token counts per line.
        # A simpler truncation from the start of history might be safer if this is too complex.
        # Let's try keeping a fixed number of recent history lines if truncation is needed.
        num_recent_history_lines_to_keep = 50 # Keep last 50 history lines (approx 50 turns)

        # Reconstruct with truncation if needed
        if len(lines) > header_end_line_idx + num_recent_history_lines_to_keep + 5: # +5 for "Current situation", "History:", "Your choice" and spacing
            header_part = "\n".join(lines[:header_end_line_idx])
            history_part_full = lines[header_end_line_idx:] # This contains "Current situation:", "History:", actual history, "Your choice:"
            
            # Find the actual "History:" line within history_part_full
            history_marker_idx_in_tail = -1
            for idx, line in enumerate(history_part_full):
                if line.strip() == "History:":
                    history_marker_idx_in_tail = idx
                    break
            
            if history_marker_idx_in_tail != -1:
                actual_history_lines = history_part_full[history_marker_idx_in_tail+1 : -1] # Exclude "Your choice:"
                if len(actual_history_lines) > num_recent_history_lines_to_keep:
                    truncated_history = actual_history_lines[-num_recent_history_lines_to_keep:]
                    prompt_text = header_part + "\n" + \
                                  "\n".join(history_part_full[:history_marker_idx_in_tail+1]) + "\n" + \
                                  "... [History Truncated] ...\n" + \
                                  "\n".join(truncated_history) + "\n" + \
                                  history_part_full[-1] # "Your choice:"
                    # print("Applied advanced truncation.")
                # Else, history isn't long enough to need this specific truncation, initial tokenizer truncation might handle it.
            # else, structure is unexpected, let Hugging Face tokenizer handle truncation.

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=effective_max_len).to(device)
    input_length = inputs.input_ids.shape[1]
        
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=10, 
            do_sample=True, 
            temperature=0.6,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True 
        )
    
    newly_generated_tokens = outputs[0, input_length:]
    generated_text = tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip()
    return generated_text

def get_llm_choice_from_response(raw_response, valid_options):
    patterns = [re.escape(str(option)) for option in valid_options] 
    regex_pattern = r'(?:^|\s|[^\w-])(' + r'|'.join(patterns) + r')(?:$|\s|[^\w-])'
    match = re.search(regex_pattern, raw_response, re.IGNORECASE)
    
    if match:
        matched_group = match.group(1)
        for option in valid_options:
            if str(option).lower() == matched_group.lower():
                return option
    
    cleaned_response_lines = [line.strip() for line in raw_response.split('\n') if line.strip()]
    if cleaned_response_lines:
        # Try to grab the first word-like token from the first non-empty line.
        # This handles cases like "STKA. Because..." or "STKA, and then..."
        first_word_match = re.match(r'([a-zA-Z0-9_]+)', cleaned_response_lines[0])
        if first_word_match:
            potential_choice_on_first_line = first_word_match.group(1)
            for option in valid_options:
                if str(option).lower() == potential_choice_on_first_line.lower():
                    return option
        # If the entire first line is an option (less likely but a fallback)
        for option in valid_options:
            if str(option).lower() == cleaned_response_lines[0].lower():
                return option
    return None # No clear match

# --- Simulation Function ---
def simulate_scenario_outcome(chosen_option_id, current_option_probabilities, outcome_scores):
    chosen_option_id_str = str(chosen_option_id)
    if chosen_option_id_str not in current_option_probabilities:
        default_bad_outcome_name = next((name for name, score in outcome_scores.items() if score < 0), None)
        if default_bad_outcome_name: return default_bad_outcome_name, outcome_scores[default_bad_outcome_name]
        default_neutral_outcome_name = next((name for name, score in outcome_scores.items() if score == 0), None)
        if default_neutral_outcome_name: return default_neutral_outcome_name, outcome_scores[default_neutral_outcome_name]
        return "SIMULATION_ERROR_CHOICE", min(outcome_scores.values()) if outcome_scores else -100

    probabilities_for_chosen_option = current_option_probabilities[chosen_option_id_str]
    possible_outcomes = list(probabilities_for_chosen_option.keys())
    outcome_probs_values = list(probabilities_for_chosen_option.values())

    sum_probs = sum(outcome_probs_values)
    if not np.isclose(sum_probs, 1.0) or np.any(np.array(outcome_probs_values) < 0):
        if sum_probs > 0 and not np.any(np.array(outcome_probs_values) < 0): 
            outcome_probs_values = np.array(outcome_probs_values) / sum_probs
            if not np.isclose(sum(outcome_probs_values), 1.0):
                 if possible_outcomes: chosen_outcome_name = possible_outcomes[0] 
                 else: return "CRITICAL_SIM_ERROR_NO_OUTCOMES_A", -200
            else: 
                 chosen_outcome_name = np.random.choice(possible_outcomes, p=outcome_probs_values)
        else: 
            if possible_outcomes: chosen_outcome_name = possible_outcomes[0] 
            else: return "CRITICAL_SIM_ERROR_NO_OUTCOMES_B", -200
    else:
        chosen_outcome_name = np.random.choice(possible_outcomes, p=outcome_probs_values)
        
    score = outcome_scores.get(chosen_outcome_name, 0)
    return chosen_outcome_name, score

# --- Helper Function to Calculate Expected Values ---
def calculate_expected_values(option_probabilities, outcome_scores):
    evs = {}
    for option, probs_dict in option_probabilities.items():
        ev = 0
        for outcome_name, prob in probs_dict.items():
            if outcome_name not in outcome_scores:
                # print(f"Warning: Outcome '{outcome_name}' for option '{option}' not found in outcome_scores. Assigning score of 0 for EV calculation.") # Verbose
                pass
            ev += prob * outcome_scores.get(outcome_name, 0)
        evs[option] = ev
    return evs

# --- Main Test Loop Function ---
def run_scenario_episode(scenario_key, current_model_id_for_run, num_iterations=25): # model_id passed for logging
    if model is None or tokenizer is None or (tokenizer and tokenizer.pad_token_id is None):
        print(f"Model/Tokenizer/PadToken not properly loaded for {current_model_id_for_run}. Skipping episode for {scenario_key}.")
        return {"total_score":0, "optimal_choice_rate":0, "valid_response_rate":0, "choices_this_episode": [], "scores_this_episode": [], "optimal_flags_this_episode": []}
    if "openai/whisper" in current_model_id_for_run:
        print(f"Skipping scenario {scenario_key} for Whisper model as it's unsuitable.")
        return {"total_score":0, "optimal_choice_rate":0, "valid_response_rate":0, "choices_this_episode": [], "scores_this_episode": [], "optimal_flags_this_episode": []}

    # print(f"--- Starting Episode for Scenario: {ALL_SCENARIOS[scenario_key]['scenario_name']} (LLM: {current_model_id_for_run}) ---") # Verbose
    
    current_scenario_config = copy.deepcopy(ALL_SCENARIOS[scenario_key])
    active_probabilities = current_scenario_config["initial_probabilities"] 
    outcome_scores = current_scenario_config["outcome_scores"]
    valid_options = current_scenario_config["options"]
    option_intro = current_scenario_config["option_intro"]
    task_description = current_scenario_config["task_description"]
    examples_for_prompt = current_scenario_config["examples_for_prompt"]
    
    previous_outputs_history = ""
    total_score_this_episode = 0
    optimal_choices_made_count = 0
    valid_responses_count = 0
    
    # For detailed logging for plot_test.py
    choices_this_episode_log = []
    scores_this_episode_log = []
    optimal_flags_this_episode_log = []


    expected_values = calculate_expected_values(active_probabilities, outcome_scores)
    optimal_option_id_ev = None
    if expected_values:
        optimal_option_id_ev = max(expected_values, key=expected_values.get)

    choice_prompt_options_str = ', '.join(map(str,valid_options))

    for i in range(num_iterations):
        iteration_num = i + 1
        
        narrative_update_text_for_llm = "" 
        prompt = f"""You are an AI agent. {task_description}
{option_intro}
Based on the history of choices and their outcomes{narrative_update_text_for_llm}, decide which option to pick next.
Output ONLY the option identifier ({choice_prompt_options_str}). Do not include any other words, explanations, or formatting.

{examples_for_prompt}

Current situation:
History:
{previous_outputs_history}Your choice ({choice_prompt_options_str}):"""

        ai_response_raw = get_llm_response(prompt)
        llm_choice = get_llm_choice_from_response(ai_response_raw, valid_options)

        history_item_prefix = "T" # Default short prefix for compactness
        if scenario_key == "StockMarket": history_item_prefix = f"Q{iteration_num}"
        elif scenario_key == "CropYield": history_item_prefix = f"S{iteration_num}"
        elif scenario_key == "PrizeDoor": history_item_prefix = f"R{iteration_num}"
        elif scenario_key == "SmugglingRoutes": history_item_prefix = f"Trip{iteration_num}"
        elif scenario_key == "DietaryAdvisor": history_item_prefix = f"C{iteration_num}"
        elif scenario_key == "SlotMachine": history_item_prefix = f"T{iteration_num}"

        current_choice_was_optimal = False
        current_score = 0

        if llm_choice is not None:
            valid_responses_count += 1
            if optimal_option_id_ev and str(llm_choice) == str(optimal_option_id_ev):
                optimal_choices_made_count += 1
                current_choice_was_optimal = True
            
            outcome_name, outcome_score = simulate_scenario_outcome(
                llm_choice, active_probabilities, outcome_scores
            )
            current_score = outcome_score
            total_score_this_episode += outcome_score
            previous_outputs_history += f"{history_item_prefix}: Chose {llm_choice}, Result: {outcome_name}\n"
        else:
            current_score = -20 # Penalty
            total_score_this_episode += current_score 
            previous_outputs_history += f"{history_item_prefix}: Chose Invalid ({ai_response_raw[:15].replace(chr(10), '')}...), Result: Penalty\n"
        
        choices_this_episode_log.append(str(llm_choice) if llm_choice is not None else "INVALID")
        scores_this_episode_log.append(current_score)
        optimal_flags_this_episode_log.append(1 if current_choice_was_optimal else 0)

        if (iteration_num % 5 == 0 or iteration_num == num_iterations) and num_iterations >= 5 :
             print(f"  {scenario_key} Ep {ep_num+1 if 'ep_num' in locals() else ''} - Iter {iteration_num}/{num_iterations}, Score: {total_score_this_episode}")


    optimal_choice_rate = optimal_choices_made_count / num_iterations if num_iterations > 0 else 0
    valid_response_rate = valid_responses_count / num_iterations if num_iterations > 0 else 0
    
    # print(f"--- Episode End for {scenario_key} (LLM: {current_model_id_for_run}) ---") # Verbose
    # print(f"  Total Score: {total_score_this_episode}, Optimal Rate: {optimal_choice_rate:.2f}, Valid Rate: {valid_response_rate:.2f}") # Verbose
    
    return {
        "total_score": total_score_this_episode, 
        "optimal_choice_rate": optimal_choice_rate, 
        "valid_response_rate": valid_response_rate,
        "choices_this_episode": choices_this_episode_log, # For detailed plotting
        "scores_this_episode": scores_this_episode_log,     # For detailed plotting
        "optimal_flags_this_episode": optimal_flags_this_episode_log # For detailed plotting
    }

# --- Main Execution ---
if __name__ == "__main__":
    if model is None or tokenizer is None or (model_id and "openai/whisper" in model_id) or \
       (tokenizer and tokenizer.pad_token_id is None and tokenizer.eos_token_id is None) :
        print("Exiting: Model/Tokenizer not suitable or not loaded properly for the chosen model (or pad_token_id issue).")
        if model_id and "openai/whisper" in model_id:
            print("Whisper model is for speech-to-text and not usable in this script.")
        exit()

    num_episodes_per_scenario = 500
    iterations_per_episode = 25
    print(f"RUN CONFIG: Model={model_id}, Episodes/Scenario={num_episodes_per_scenario}, Iterations/Episode={iterations_per_episode}")

    results_dir = "experiment_results_data" # Directory to save raw data for plotting
    os.makedirs(results_dir, exist_ok=True)
    
    print("\nAvailable Scenarios:")
    scenario_keys_list = list(ALL_SCENARIOS.keys())
    for i, key in enumerate(scenario_keys_list):
        print(f" ({i+1}) {key} ({ALL_SCENARIOS[key]['scenario_name']})")
    print(f" (A) All {len(scenario_keys_list)} Scenarios")
    
    scenario_selection_input = input(f"Choose scenario number (1-{len(scenario_keys_list)}) or 'A' for All: ").strip().upper()

    scenarios_to_run_keys = []
    if scenario_selection_input == 'A':
        scenarios_to_run_keys = scenario_keys_list
    else:
        try:
            choice_idx = int(scenario_selection_input) - 1
            if 0 <= choice_idx < len(scenario_keys_list):
                scenarios_to_run_keys.append(scenario_keys_list[choice_idx])
            else:
                print("Invalid scenario number. Exiting.")
                exit()
        except ValueError:
            print("Invalid input. Exiting.")
            exit()

    experiment_timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_model_name_for_file = model_id.replace("/", "_").replace("-","_") # Make filename safer

    overall_summary_for_console = {}


    for scenario_id_key in scenarios_to_run_keys:
        print(f"\n\n================== RUNNING SCENARIO: {ALL_SCENARIOS[scenario_id_key]['scenario_name']} ==================")
        
        # Data lists for this specific scenario run (model + scenario)
        all_episodes_optimal_choice_ratios = []
        all_episodes_total_scores = []
        # If you want to plot average learning curves, you'll need to store all turn-by-turn data from all episodes
        # For now, plot_test.py expects data from *one* scenario-model run (i.e. results for 500 episodes)

        for ep_num in range(num_episodes_per_scenario):
            # print(f"--- Episode {ep_num+1}/{num_episodes_per_scenario} for {scenario_id_key} ---") # Verbose
            # Pass the current model_id to the run_scenario_episode function
            episode_stats = run_scenario_episode(scenario_id_key, model_id, iterations_per_episode)
            
            all_episodes_optimal_choice_ratios.append(episode_stats["optimal_choice_rate"])
            all_episodes_total_scores.append(episode_stats["total_score"])
            # `episode_stats` also contains "choices_this_episode", "scores_this_episode", "optimal_flags_this_episode"
            # if you wanted to save ALL raw data for complex plotting later.
            # For the current plot_test.py to match the image, we only need the final per-episode stats.
        
        # Save the per-episode data for this scenario and model to a JSON file
        # This is the data plot_test.py will use
        scenario_results_data_for_plotting = {
            "model_name": model_id,
            "scenario_id": scenario_id_key,
            "scenario_name": ALL_SCENARIOS[scenario_id_key]['scenario_name'],
            "optimal_choice_ratios_per_episode": all_episodes_optimal_choice_ratios, 
            "total_scores_per_episode": all_episodes_total_scores, 
            "num_episodes": num_episodes_per_scenario,
            "iterations_per_episode": iterations_per_episode,
            "timestamp": experiment_timestamp
        }
        
        results_filename = os.path.join(results_dir, f"plotdata_{safe_model_name_for_file}_{scenario_id_key}_{experiment_timestamp}.json")
        with open(results_filename, "w") as f:
            json.dump(scenario_results_data_for_plotting, f, indent=4)
        print(f"\nSaved data for plotting {scenario_id_key} (LLM: {model_id}) to: {results_filename}")
        
        # For console summary
        mean_score = np.mean(all_episodes_total_scores) if all_episodes_total_scores else 0
        std_score = np.std(all_episodes_total_scores) if all_episodes_total_scores else 0
        mean_optimal_rate = np.mean(all_episodes_optimal_choice_ratios) if all_episodes_optimal_choice_ratios else 0
        
        overall_summary_for_console[scenario_id_key] = {
            "mean_total_score_per_episode": mean_score,
            "std_total_score_per_episode": std_score,
            "mean_optimal_choice_rate": mean_optimal_rate,
        }
        print(f"  Mean Total Score across {num_episodes_per_scenario} episodes: {mean_score:.2f} (Std: {std_score:.2f})")
        print(f"  Mean Optimal Choice Rate across {num_episodes_per_scenario} episodes: {mean_optimal_rate:.2f}")


    print("\n\n=============== CONSOLE SUMMARY FOR THIS RUN ===============")
    print(f"LLM Used: {model_id}")
    print(f"Episodes per Scenario: {num_episodes_per_scenario}, Iterations per Episode: {iterations_per_episode}")
    for scenario_id_key, results in overall_summary_for_console.items():
        print(f"\nScenario: {ALL_SCENARIOS[scenario_id_key]['scenario_name']}")
        print(f"  Mean Total Score per Episode: {results['mean_total_score_per_episode']:.2f} (Std: {results['std_total_score_per_episode']:.2f})")
        print(f"  Mean Optimal Choice Rate (picked EV-best based on initial probabilities): {results['mean_optimal_choice_rate']:.2f}")
    print(f"\nData for detailed plots saved in individual files in the '{results_dir}' directory.")
