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

# --- Scenario Definitions (Same as before) ---
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
    '3': "meta-llama/Llama-3.1-8B",
    '4': "deepseek-ai/DeepSeek-R1",
    '5': "microsoft/phi-2",
    '6': "google/gemma-3-12b-it",
    '7': "openai/whisper-large-v3"
}

print("Please select the LLM model (using a number from your original list):")
print(" (1) Qwen/Qwen3-4B (Note: May require 'trust_remote_code=True')")
print(" (2) Qwen/Qwen3-8B (Note: May require 'trust_remote_code=True')")
print(" (3) meta-llama/Llama-3.1-8B (Note: Base Llama models might need specific prompt formatting or fine-tuning for optimal instruction following. An 'Instruct' or 'Chat' variant is usually recommended for these tasks.)")
print(" (4) deepseek-ai/DeepSeek-R1 (Note: May require 'trust_remote_code=True')")
print(" (5) microsoft/phi-2 (Note: May require 'trust_remote_code=True')")
print(" (6) google/gemma-3-12b-it (WARNING: 'gemma-3-12b-it' is not a recognized standard public Hugging Face model ID. This will likely fail to load unless it's a private model or a typo for a Gemma/Gemma2 model like 'google/gemma-2-9b-it'.)")
print(" (7) openai/whisper-large-v3 (WARNING: This is a SPEECH-TO-TEXT model and is NOT suitable for this text generation task. It will likely cause errors or produce unusable output.)")

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
    elif "gemma-3" in model_id.lower() and not os.path.exists(os.path.join(".", model_id.split("/")[-1] if "/" in model_id else model_id)): # Basic check if it's not a local path
        print(f"WARNING: Model ID '{model_id}' appears to be non-standard and might fail to load from Hugging Face Hub. If it's a private model, ensure it's accessible. Attempting to load anyway.")
    
    if model is None and "openai/whisper" not in model_id :
        try:
            if device.type == "mps": dtype = torch.float16
            elif device.type == "cuda": dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            else: dtype = torch.float32
            
            print(f"Loading model {model_id} with dtype: {dtype}")
            model_load_args = {"torch_dtype": dtype, "cache_dir": "."}
            
            # Models that often require trust_remote_code
            models_needing_trust = ["microsoft/phi-2", "deepseek-ai/DeepSeek-R1", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]
            if model_id in models_needing_trust or "deepseek" in model_id.lower() or "qwen3" in model_id.lower() :
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
            if "trust_remote_code" not in model_load_args and (model_id in models_needing_trust or "deepseek" in model_id.lower() or "qwen3" in model_id.lower()):
                print(f"Hint: Model {model_id} might require 'trust_remote_code=True' during loading.")
            if "gemma-3" in model_id.lower():
                 print(f"Hint: Loading for '{model_id}' failed. This might be due to an incorrect ID. Standard Gemma models are 'google/gemma-2-...' or 'google/gemma-...'.")


if tokenizer and tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
        print("Tokenizer does not have a pad_token_id. Setting it to eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        print("Warning: Tokenizer has no pad_token_id and no eos_token_id. Adding a new pad_token '[PAD]'.")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if model: 
             try: # Resize embeddings only if model is a type that supports it
                model.resize_token_embeddings(len(tokenizer))
             except AttributeError:
                print(f"Could not resize token embeddings for model type {type(model)}. This might be fine or might cause issues if the new pad token is used.")


# --- LLM Interaction Functions ---
def get_llm_response(prompt_text): # The function signature stays the same
    if model is None or tokenizer is None:
        return "ERROR_MODEL_NOT_LOADED"
    if tokenizer.pad_token_id is None: 
        return "ERROR_PAD_TOKEN_NONE"
    if "openai/whisper" in model_id:
        return "ERROR_WRONG_MODEL_TYPE_FOR_TASK"

    # --- Start of New Code ---
    # The 'prompt_text' we receive is the full block of text. We need to parse it
    # to build the structured message list that apply_chat_template needs.

    # 1. Separate the system prompt from the user prompt part.
    # The system prompt is everything before "Current situation:".
    try:
        system_prompt_part, user_prompt_part = prompt_text.split("Current situation:", 1)
        system_prompt_part = system_prompt_part.strip()
        user_prompt_part = "Current situation:" + user_prompt_part # Add the header back
    except ValueError:
        # Fallback if the split string isn't found
        system_prompt_part = "You are an AI agent making a choice based on history."
        user_prompt_part = prompt_text

    # 2. Construct the message list
    messages = [
        {"role": "system", "content": system_prompt_part},
        {"role": "user", "content": user_prompt_part.strip()}
    ]

    # 3. Apply the chat template
    # This is the magic line that formats the prompt correctly for the model.
    # We set tokenize=False because we want the formatted string, not the tokens yet.
    # add_generation_prompt=True adds the special tokens to signal the assistant should start talking.
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # --- End of New Code ---

    # The rest of the function proceeds as before, but using `formatted_prompt`
    effective_max_len = tokenizer.model_max_length - 60 if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length else 2048 - 60
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=effective_max_len).to(device)
    input_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=50, 
            do_sample=True, 
            temperature=0.6,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    newly_generated_tokens = outputs[0, input_length:]
    generated_text = tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip()
    return generated_text

def get_llm_choice_from_response(raw_response, valid_options):
    # Clean the response by stripping leading/trailing whitespace
    cleaned_response = raw_response.strip()

    # First, check if any line is an exact match for an option (very common)
    for line in cleaned_response.split('\n'):
        # Clean each line as well
        potential_choice = line.strip()
        for option in valid_options:
            if str(option).lower() == potential_choice.lower():
                return option # Found an exact match on a line

    # If no exact line match, search for the option as a whole word anywhere in the response
    # This is very forgiving and handles cases like "My choice is SLOT2."
    for option in valid_options:
        # The \b ensures we match whole words only (so 'CROP' doesn't match 'CROPX')
        pattern = r'\b' + re.escape(str(option)) + r'\b'
        match = re.search(pattern, cleaned_response, re.IGNORECASE)
        if match:
            return option # Found the first valid option in the text

    # If we still haven't found anything, return None
    return None

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
                pass
            ev += prob * outcome_scores.get(outcome_name, 0)
        evs[option] = ev
    return evs

# --- Main Test Loop Function ---
def run_scenario_episode(scenario_key_arg, current_model_id_for_run_arg, current_run_num_arg, total_runs_arg, num_turns=25):
    if model is None or tokenizer is None or (tokenizer and tokenizer.pad_token_id is None):
        return {"total_score":0, "optimal_choice_rate":0, "valid_response_rate":0, "choices_this_run": [], "scores_this_run": [], "optimal_flags_this_run": []}
    if "openai/whisper" in current_model_id_for_run_arg:
        return {"total_score":0, "optimal_choice_rate":0, "valid_response_rate":0, "choices_this_run": [], "scores_this_run": [], "optimal_flags_this_run": []}
    
    current_scenario_config = copy.deepcopy(ALL_SCENARIOS[scenario_key_arg])
    active_probabilities = current_scenario_config["initial_probabilities"] 
    outcome_scores = current_scenario_config["outcome_scores"]
    valid_options = current_scenario_config["options"]
    option_intro = current_scenario_config["option_intro"]
    task_description = current_scenario_config["task_description"]
    examples_for_prompt = current_scenario_config["examples_for_prompt"]
    
    previous_outputs_history = ""
    total_score_this_run = 0
    optimal_choices_made_count = 0
    valid_responses_count = 0
    
    choices_this_run_log = []
    scores_this_run_log = []
    optimal_flags_this_run_log = []

    expected_values = calculate_expected_values(active_probabilities, outcome_scores)
    optimal_option_id_ev = None
    if expected_values:
        optimal_option_id_ev = max(expected_values, key=expected_values.get)

    choice_prompt_options_str = ', '.join(map(str,valid_options))

    # Print progress for the start of a new "run" (what was previously episode)
    print(f"  Scenario: {ALL_SCENARIOS[scenario_key_arg]['scenario_name']} - Starting Run {current_run_num_arg}/{total_runs_arg} (Model: {current_model_id_for_run_arg})")

    for i in range(num_turns):
        turn_num = i + 1
        
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

        history_item_prefix = f"T{turn_num}" 
        if scenario_key_arg == "StockMarket": history_item_prefix = f"Q{turn_num}"
        elif scenario_key_arg == "CropYield": history_item_prefix = f"S{turn_num}"
        elif scenario_key_arg == "PrizeDoor": history_item_prefix = f"R{turn_num}"
        elif scenario_key_arg == "SmugglingRoutes": history_item_prefix = f"Trip{turn_num}"
        elif scenario_key_arg == "DietaryAdvisor": history_item_prefix = f"C{turn_num}"
        # SlotMachine will use the default "T{turn_num}"

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
            total_score_this_run += outcome_score
            previous_outputs_history += f"{history_item_prefix}: Chose {llm_choice}, Result: {outcome_name}\n"
        else:
            current_score = -20 
            total_score_this_run += current_score 
            print(f"\n>>>> DEBUG: Invalid raw response was: '{ai_response_raw}' <<<<\n")
            previous_outputs_history += f"{history_item_prefix}: Chose Invalid ({ai_response_raw[:15].replace(chr(10), '')}...), Result: Penalty\n"

        
        choices_this_run_log.append(str(llm_choice) if llm_choice is not None else "INVALID")
        scores_this_run_log.append(current_score)
        optimal_flags_this_run_log.append(1 if current_choice_was_optimal else 0)
        
        # Reduced frequency of printing for cleaner long logs
        if num_turns >=10 and (turn_num % (num_turns // 2) == 0 or turn_num == num_turns) and turn_num > 0:
            print(f"    Run {current_run_num_arg} - Turn {turn_num}/{num_turns}, Current Total Score: {total_score_this_run}")


    optimal_choice_rate = optimal_choices_made_count / num_turns if num_turns > 0 else 0
    valid_response_rate = valid_responses_count / num_turns if num_turns > 0 else 0
    
    return {
        "total_score_this_run": total_score_this_run, 
        "optimal_choice_rate_this_run": optimal_choice_rate, 
        "valid_response_rate_this_run": valid_response_rate,
        # Data for plot_test.py (though plot_test.py currently uses the per-run summaries, not per-turn)
        # "choices_per_turn_in_run": choices_this_run_log, 
        # "scores_per_turn_in_run": scores_this_run_log,     
        # "optimal_flags_per_turn_in_run": optimal_flags_this_run_log 
    }

# --- Main Execution ---
if __name__ == "__main__":
    if model is None or tokenizer is None or (model_id and "openai/whisper" in model_id) or \
       (tokenizer and tokenizer.pad_token_id is None and tokenizer.eos_token_id is None and model_id != "microsoft/phi-2") :
        print("Exiting: Model/Tokenizer not suitable or not loaded properly for the chosen model (or pad_token_id issue).")
        if model_id and "openai/whisper" in model_id:
            print("Whisper model is for speech-to-text and not usable in this script.")
        exit()

    # Hardcoded to your 500x25 structure (Run = outer loop, Turn = inner loop)
    num_runs_per_scenario = 500
    num_turns_per_run = 25
    print(f"RUN CONFIG: Model={model_id}, Runs/Scenario={num_runs_per_scenario}, Turns/Run={num_turns_per_run}")

    results_dir = "experiment_results_data" 
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
    safe_model_name_for_file = model_id.replace("/", "_").replace("-","_") 

    overall_summary_for_console = {} # For printing a summary at the very end

    for scenario_id_key in scenarios_to_run_keys:
        print(f"\n\n================== PROCESSING SCENARIO: {ALL_SCENARIOS[scenario_id_key]['scenario_name']} ==================")
        
        # Lists to store results for each run (sampling iteration) for the current scenario
        list_of_optimal_choice_ratios_for_this_scenario = []
        list_of_total_scores_for_this_scenario = []
        list_of_valid_response_rates_for_this_scenario = []

        for run_idx in range(num_runs_per_scenario):
            run_stats = run_scenario_episode(scenario_id_key, model_id, run_idx + 1, num_runs_per_scenario, num_turns_per_run)
            
            list_of_optimal_choice_ratios_for_this_scenario.append(run_stats["optimal_choice_rate_this_run"])
            list_of_total_scores_for_this_scenario.append(run_stats["total_score_this_run"])
            list_of_valid_response_rates_for_this_scenario.append(run_stats["valid_response_rate_this_run"])
        
        # Save the per-run data for this scenario and model to a JSON file for plot_test.py
        scenario_results_data_for_plotting = {
            "model_name": model_id,
            "scenario_id": scenario_id_key,
            "scenario_name": ALL_SCENARIOS[scenario_id_key]['scenario_name'],
            "optimal_choice_ratios_per_run": list_of_optimal_choice_ratios_for_this_scenario, 
            "total_scores_per_run": list_of_total_scores_for_this_scenario, 
            "valid_response_rates_per_run": list_of_valid_response_rates_for_this_scenario,
            "num_runs": num_runs_per_scenario, # Formerly num_episodes
            "num_turns_per_run": num_turns_per_run, # Formerly iterations_per_episode
            "timestamp": experiment_timestamp
        }
        
        results_filename = os.path.join(results_dir, f"plotdata_{safe_model_name_for_file}_{scenario_id_key}_{experiment_timestamp}.json")
        with open(results_filename, "w") as f:
            json.dump(scenario_results_data_for_plotting, f, indent=4)
        print(f"\nSaved data for plotting {scenario_id_key} (LLM: {model_id}) to: {results_filename}")
        
        # Store for overall console summary
        mean_score = np.mean(list_of_total_scores_for_this_scenario) if list_of_total_scores_for_this_scenario else 0
        std_score = np.std(list_of_total_scores_for_this_scenario) if list_of_total_scores_for_this_scenario else 0
        mean_optimal_rate = np.mean(list_of_optimal_choice_ratios_for_this_scenario) if list_of_optimal_choice_ratios_for_this_scenario else 0
        mean_valid_rate = np.mean(list_of_valid_response_rates_for_this_scenario) if list_of_valid_response_rates_for_this_scenario else 0
        
        overall_summary_for_console[scenario_id_key] = {
            "mean_total_score_per_run": mean_score,
            "std_total_score_per_run": std_score,
            "mean_optimal_choice_rate_per_run": mean_optimal_rate,
            "mean_valid_response_rate_per_run": mean_valid_rate
        }
        print(f"  Mean Total Score across {num_runs_per_scenario} runs: {mean_score:.2f} (Std: {std_score:.2f})")
        print(f"  Mean Optimal Choice Rate across {num_runs_per_scenario} runs: {mean_optimal_rate:.2f}")
        print(f"  Mean Valid Response Rate across {num_runs_per_scenario} runs: {mean_valid_rate:.2f}")


    print("\n\n=============== FINAL CONSOLE SUMMARY FOR THIS LLM ===============")
    print(f"LLM Used: {model_id}")
    print(f"Runs per Scenario: {num_runs_per_scenario}, Turns per Run: {num_turns_per_run}")
    for scenario_id_key, results in overall_summary_for_console.items():
        print(f"\nScenario: {ALL_SCENARIOS[scenario_id_key]['scenario_name']}")
        print(f"  Mean Total Score per Run: {results['mean_total_score_per_run']:.2f} (Std: {results['std_total_score_per_run']:.2f})")
        print(f"  Mean Optimal Choice Rate per Run (picked EV-best): {results['mean_optimal_choice_rate_per_run']:.2f}")
        print(f"  Mean Valid Response Rate per Run: {results['mean_valid_response_rate_per_run']:.2f}")
    print(f"\nData for detailed plots saved in individual JSON files in the '{results_dir}' directory.")
    print("Each JSON file can be used as input for plot_test.py")
