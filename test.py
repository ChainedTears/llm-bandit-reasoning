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

# --- Scenario Definitions (Same as before, with SlotMachine included) ---
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

# --- Model Setup (Using YOUR original list) ---
# This is your requested model dictionary
model_dict = {
    '1': "Qwen/Qwen2-1.5B-Instruct", # My suggestion for Qwen3-4B
    '2': "Qwen/Qwen2-7B-Instruct",  # My suggestion for Qwen3-8B
    '3': "meta-llama/Meta-Llama-3-8B-Instruct", # My suggestion for Llama-3.1-8B
    '4': "deepseek-ai/DeepSeek-R1", # Kept as per your request. NOTE: May require trust_remote_code=True
    '5': "microsoft/phi-2", # Kept as per your request
    '6': "google/gemma-2-9b-it", # My suggestion for gemma-3-12b-it
    '7': "openai/whisper-large-v3"  # Kept as per your request. WARNING: This is a SPEECH-TO-TEXT model.
}

print("Please select the LLM model (using a number from your original list, with some suggestions):")
print(" (1) Qwen/Qwen2-1.5B-Instruct (Suggested for Qwen3-4B)")
print(" (2) Qwen/Qwen2-7B-Instruct (Suggested for Qwen3-8B)")
print(" (3) meta-llama/Meta-Llama-3-8B-Instruct (Suggested for Llama-3.1-8B)")
print(" (4) deepseek-ai/DeepSeek-R1")
print(" (5) microsoft/phi-2")
print(" (6) google/gemma-2-9b-it (Suggested for gemma-3-12b-it)")
print(" (7) openai/whisper-large-v3 (WARNING: Speech-to-text model! Unlikely to work for this task.)")

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

# Load model
model = None
if tokenizer:
    if "openai/whisper" in model_id: # Prevent trying to load Whisper with AutoModelForCausalLM
        print(f"ERROR: {model_id} is a Whisper (speech-to-text) model and cannot be loaded with AutoModelForCausalLM for this text generation task. Please choose a generative text model.")
    else:
        try:
            if device.type == "mps": dtype = torch.float16
            elif device.type == "cuda": dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            else: dtype = torch.float32
            
            print(f"Loading model {model_id} with dtype: {dtype}")
            model_load_args = {"torch_dtype": dtype, "cache_dir": "."}
            
            if model_id == "microsoft/phi-2" or "deepseek" in model_id:
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
            if "trust_remote_code" not in model_load_args and ("phi-2" in model_id or "deepseek" in model_id):
                print("Hint: This model might require 'trust_remote_code=True' during loading.")

if tokenizer and tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
        print("Tokenizer does not have a pad_token_id. Setting it to eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        print("Warning: Tokenizer has no pad_token_id and no eos_token_id. Adding a new pad_token '[PAD]'. This might require model embedding resizing if model is already loaded.")
        # This part is tricky if model is already loaded; ideally, handle before model load or ensure models used have it.
        # For simplicity here, we just set it on tokenizer. If errors occur, resizing model embeddings might be needed.
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if model:
             model.resize_token_embeddings(len(tokenizer))


# --- LLM Interaction Functions ---
def get_llm_response(prompt_text):
    if model is None or tokenizer is None:
        # print("Model or tokenizer not loaded. Cannot get response.")
        return "ERROR_MODEL_NOT_LOADED"
    if tokenizer.pad_token_id is None: 
        # print("Error: pad_token_id is None. Cannot generate response.")
        return "ERROR_PAD_TOKEN_NONE"
    if "openai/whisper" in model_id:
        # print("Error: Cannot use Whisper model for text generation in this task.")
        return "ERROR_WRONG_MODEL_TYPE"

    prompt_max_len = tokenizer.model_max_length - 60 # Reserve 60 tokens for generation and special tokens
    current_prompt_tokens = tokenizer.encode(prompt_text)
    
    if len(current_prompt_tokens) > prompt_max_len:
        # print(f"Warning: Prompt too long ({len(current_prompt_tokens)} tokens), truncating from the beginning.")
        # A simple truncation strategy: try to preserve the initial instruction block and the most recent history.
        # This needs to be done carefully to not break the prompt structure.
        # Split prompt into lines
        lines = prompt_text.split('\n')
        # Heuristic: preserve first few lines (instructions, examples), and last few lines (current history, choice line)
        # This assumes your examples and main instructions are relatively short.
        num_header_lines = 0
        # Count lines until "Example 1:" or "Current situation:" to estimate header
        for idx, line in enumerate(lines):
            if "Example 1:" in line or "Current situation:" in line :
                num_header_lines = idx
                break
        if num_header_lines == 0 and len(lines) > 20: num_header_lines = 10 # Fallback if structure is unexpected

        num_tail_lines = 15 # Estimate for "Current situation:", History, and "Your choice:"
        
        if len(lines) > num_header_lines + num_tail_lines :
            truncated_history_indicator = "\n... [History Truncated] ...\n"
            # Keep header, indicator, and tail
            prompt_text = "\n".join(lines[:num_header_lines]) + \
                           truncated_history_indicator + \
                           "\n".join(lines[-(num_tail_lines):])
            # print(f"   New truncated prompt (approx lines: {num_header_lines + 1 + num_tail_lines}):\n{prompt_text[:300]}...\n...{prompt_text[-300:]}")
    
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=False).to(device) # Final check with tokenizer's own logic
    
    # If still too long after our manual attempt, let transformers handle it with its own truncation.
    # This is a safeguard. The manual truncation should ideally prevent this.
    if inputs.input_ids.shape[1] > tokenizer.model_max_length - 10:
         print(f"Warning: Input still too long ({inputs.input_ids.shape[1]}), applying transformers truncation.")
         inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length - 10).to(device)

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
            early_stopping=True # Stop if EOS is generated
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
        first_potential_choice = cleaned_response_lines[0].split(' ')[0].strip().split(',')[0].strip().split('.')[0].strip()
        for option in valid_options:
            if str(option).lower() == first_potential_choice.lower():
                return option
        for option in valid_options: # Check if the entire first line is an option
            if str(option).lower() == cleaned_response_lines[0].lower():
                return option
                
    # print(f"Warning: LLM response '{raw_response}' did not clearly match valid options: {valid_options}.") # Verbose
    return None

# --- Simulation Function ---
def simulate_scenario_outcome(chosen_option_id, current_option_probabilities, outcome_scores):
    chosen_option_id_str = str(chosen_option_id)
    if chosen_option_id_str not in current_option_probabilities:
        # print(f"Error: Option '{chosen_option_id_str}' not found in current probabilities for simulation.") # Verbose
        # Simplified error handling: return a default bad outcome if possible
        default_bad_outcome_name = next((name for name, score in outcome_scores.items() if score < 0), None)
        if default_bad_outcome_name:
            return default_bad_outcome_name, outcome_scores[default_bad_outcome_name]
        # If no negatively scored outcome, return a neutral or arbitrary bad one
        default_neutral_outcome_name = next((name for name, score in outcome_scores.items() if score == 0), None)
        if default_neutral_outcome_name:
            return default_neutral_outcome_name, outcome_scores[default_neutral_outcome_name]
        return "SIMULATION_ERROR_CHOICE", min(outcome_scores.values()) if outcome_scores else -100


    probabilities_for_chosen_option = current_option_probabilities[chosen_option_id_str]
    possible_outcomes = list(probabilities_for_chosen_option.keys())
    outcome_probs_values = list(probabilities_for_chosen_option.values())

    sum_probs = sum(outcome_probs_values)
    if not np.isclose(sum_probs, 1.0) or np.any(np.array(outcome_probs_values) < 0):
        # print(f"Warning/Error: Probabilities for {chosen_option_id_str} are invalid (sum: {sum_probs}, values: {outcome_probs_values}). Normalizing.") # Verbose
        if sum_probs > 0 and not np.any(np.array(outcome_probs_values) < 0): 
            outcome_probs_values = np.array(outcome_probs_values) / sum_probs
            if not np.isclose(sum(outcome_probs_values), 1.0): # Should be close after normalization
                 if possible_outcomes: chosen_outcome_name = possible_outcomes[0] 
                 else: return "CRITICAL_SIM_ERROR_NO_OUTCOMES_A", -200 # Should not happen
            else: # Normalization successful
                 chosen_outcome_name = np.random.choice(possible_outcomes, p=outcome_probs_values)
        else: 
            if possible_outcomes: chosen_outcome_name = possible_outcomes[0] 
            else: return "CRITICAL_SIM_ERROR_NO_OUTCOMES_B", -200 # Should not happen
    else:
        chosen_outcome_name = np.random.choice(possible_outcomes, p=outcome_probs_values)
        
    score = outcome_scores.get(chosen_outcome_name, 0)
    return chosen_outcome_name, score

# --- Helper Function to Calculate Expected Values ---
def calculate_expected_values(option_probabilities, outcome_scores):
    evs = {}
    for option, probs_dict in option_probabilities.items():
        ev = 0
        # Ensure all outcomes in probs_dict exist in outcome_scores, or handle missing ones
        for outcome_name, prob in probs_dict.items():
            if outcome_name not in outcome_scores:
                print(f"Warning: Outcome '{outcome_name}' for option '{option}' not found in outcome_scores. Assigning score of 0 for EV calculation.")
            ev += prob * outcome_scores.get(outcome_name, 0)
        evs[option] = ev
    return evs

# --- Main Test Loop Function ---
def run_scenario_episode(scenario_key, current_model_id, num_iterations=25):
    if model is None or tokenizer is None or (tokenizer and tokenizer.pad_token_id is None):
        print(f"Model/Tokenizer/PadToken not properly loaded for {current_model_id}. Skipping episode for {scenario_key}.")
        return {"total_score":0, "optimal_choice_rate":0, "valid_response_rate":0}
    if "openai/whisper" in current_model_id:
        print(f"Skipping scenario {scenario_key} for Whisper model as it's unsuitable.")
        return {"total_score":0, "optimal_choice_rate":0, "valid_response_rate":0}

    print(f"--- Starting Episode for Scenario: {ALL_SCENARIOS[scenario_key]['scenario_name']} (LLM: {current_model_id}) ---")
    
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

    expected_values = calculate_expected_values(active_probabilities, outcome_scores)
    optimal_option_id_ev = None
    if expected_values:
        optimal_option_id_ev = max(expected_values, key=expected_values.get)

    choice_prompt_options_str = ', '.join(map(str,valid_options)) # Ensure options are strings

    for i in range(num_iterations):
        iteration_num = i + 1
        
        # Construct the prompt (no narrative updates in this version)
        narrative_update_text_for_llm = "" # This is now always empty
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

        history_item_prefix = "Turn" 
        if scenario_key == "StockMarket": history_item_prefix = f"Q{iteration_num}"
        elif scenario_key == "CropYield": history_item_prefix = f"Season {iteration_num}"
        elif scenario_key == "PrizeDoor": history_item_prefix = f"Round {iteration_num}"
        elif scenario_key == "SmugglingRoutes": history_item_prefix = f"Trip {iteration_num}"
        elif scenario_key == "DietaryAdvisor": history_item_prefix = f"Client{iteration_num}"
        elif scenario_key == "SlotMachine": history_item_prefix = f"Turn {iteration_num}"

        if llm_choice is not None:
            valid_responses_count += 1
            if optimal_option_id_ev and str(llm_choice) == str(optimal_option_id_ev):
                optimal_choices_made_count += 1
            
            outcome_name, outcome_score = simulate_scenario_outcome(
                llm_choice, active_probabilities, outcome_scores
            )
            total_score_this_episode += outcome_score
            previous_outputs_history += f"{history_item_prefix}: Chose {llm_choice}, Result: {outcome_name}\n"
        else:
            total_score_this_episode -= 20 
            previous_outputs_history += f"{history_item_prefix}: Chose Invalid ({ai_response_raw[:20]}...), Result: Penalty Applied\n"
        
        # Print progress every few iterations
        if (iteration_num % 5 == 0 or iteration_num == num_iterations) and num_iterations > 5 :
            print(f"  {scenario_key} Ep. Progress: Iteration {iteration_num}/{num_iterations}, Current Total Score: {total_score_this_episode}")


    optimal_choice_rate = optimal_choices_made_count / num_iterations if num_iterations > 0 else 0
    valid_response_rate = valid_responses_count / num_iterations if num_iterations > 0 else 0
    
    print(f"--- Episode End for {scenario_key} (LLM: {current_model_id}) ---")
    print(f"  Total Score for this Episode: {total_score_this_episode}")
    print(f"  Optimal Choice Rate for this Episode (EV-based): {optimal_choice_rate:.2f}")
    print(f"  Valid Response Rate for this Episode: {valid_response_rate:.2f}")
    print("----------------------------------------------------------")
    
    return {
        "total_score": total_score_this_episode, # This is the "Cumulative Reward" for the episode
        "optimal_choice_rate": optimal_choice_rate, # This is the "Accuracy Ratio" for the episode
        "valid_response_rate": valid_response_rate
    }

# --- Main Execution ---
if __name__ == "__main__":
    if model is None or tokenizer is None or (model_id and "openai/whisper" in model_id) or \
       (tokenizer and tokenizer.pad_token_id is None and tokenizer.eos_token_id is None) : # Added check for whisper here too
        print("Exiting: Model/Tokenizer not suitable or not loaded properly for the chosen model (or pad_token_id issue).")
        if model_id and "openai/whisper" in model_id:
            print("Whisper model is for speech-to-text and not usable in this script.")
        exit()

    # Hardcoded to your 500x25 structure
    num_episodes_per_scenario = 500
    iterations_per_episode = 25
    print(f"Running with {num_episodes_per_scenario} episodes per scenario, and {iterations_per_episode} iterations per episode.")

    # Create a directory for results if it doesn't exist
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)
    
    print("\nAvailable Scenarios:")
    scenario_keys_list = list(ALL_SCENARIOS.keys())
    for i, key in enumerate(scenario_keys_list):
        print(f" ({i+1}) {key}")
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
    safe_model_name = model_id.replace("/", "_").replace("-","_") # Make filename safer

    for scenario_id_key in scenarios_to_run_keys:
        print(f"\n\n================== RUNNING SCENARIO: {ALL_SCENARIOS[scenario_id_key]['scenario_name']} ==================")
        
        # Data for plot_test.py for this specific scenario run
        per_episode_optimal_choice_ratios = []
        per_episode_total_scores = []
        # Also good to track valid response rates per episode if you want to plot that
        per_episode_valid_response_rates = []


        for ep_num in range(num_episodes_per_scenario):
            # print(f"--- Episode {ep_num+1}/{num_episodes_per_scenario} for {scenario_id_key} ---") # Less verbose overall
            episode_stats = run_scenario_episode(scenario_id_key, model_id, iterations_per_episode)
            
            per_episode_optimal_choice_ratios.append(episode_stats["optimal_choice_rate"])
            per_episode_total_scores.append(episode_stats["total_score"])
            per_episode_valid_response_rates.append(episode_stats["valid_response_rate"])
        
        # Save the per-episode data for this scenario and model to a JSON file
        scenario_results_data_for_plotting = {
            "model_name": model_id,
            "scenario_id": scenario_id_key,
            "scenario_name": ALL_SCENARIOS[scenario_id_key]['scenario_name'],
            "optimal_choice_ratios_per_episode": per_episode_optimal_choice_ratios, # This is your "Accuracy Ratio" list
            "total_scores_per_episode": per_episode_total_scores, # This is your "Cumulative Reward" (per episode) list
            "valid_response_rates_per_episode": per_episode_valid_response_rates,
            "num_episodes": num_episodes_per_scenario,
            "iterations_per_episode": iterations_per_episode,
            "timestamp": experiment_timestamp
        }
        
        results_filename = os.path.join(results_dir, f"plotdata_{safe_model_name}_{scenario_id_key}_{experiment_timestamp}.json")
        with open(results_filename, "w") as f:
            json.dump(scenario_results_data_for_plotting, f, indent=4)
        print(f"\nSaved per-episode data for {scenario_id_key} (LLM: {model_id}) to: {results_filename}")
        print(f"  Mean Total Score across {num_episodes_per_scenario} episodes: {np.mean(per_episode_total_scores):.2f}")
        print(f"  Mean Optimal Choice Rate across {num_episodes_per_scenario} episodes: {np.mean(per_episode_optimal_choice_ratios):.2f}")

    print("\n\n=============== EXPERIMENT COMPLETE ===============")
    print(f"Data for plotting saved in individual files in the '{results_dir}' directory.")
