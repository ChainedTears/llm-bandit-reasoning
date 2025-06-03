import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import secrets
# from huggingface_hub import login # You'll need to be logged in via CLI or uncomment and use your token
import re
import numpy as np
import copy # For deep copying scenario configurations

# --- Hugging Face Login (ensure you're logged in, or use a token) ---
# login(token="YOUR_HF_TOKEN_HERE") # Replace if you're not using CLI login

# --- Scenario Definitions ---
ALL_SCENARIOS = {
    "SlotMachine": {
        "scenario_name": "Classic Slot Machine Challenge",
        "task_description": "Your task is to choose between playing SLOT1 or SLOT2.",
        "option_intro": "Each slot machine can result in a 'WIN' or a 'LOSS'. Their win rates are unknown to you initially.",
        "options": ["SLOT1", "SLOT2"], # Using more descriptive IDs
        "outcome_names": ["WIN", "LOSS"],
        "outcome_scores": {"WIN": 1, "LOSS": 0}, # Classic win/loss scoring
        "initial_probabilities": { # Hidden from LLM
            "SLOT1": {"WIN": 0.30, "LOSS": 0.70}, # Arm 1: 30% win
            "SLOT2": {"WIN": 0.65, "LOSS": 0.35}  # Arm 2: 65% win (objectively better)
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

# --- Model Setup (mostly from your script) ---
# Specify model ID
model_choices_display = {
    '1': "Qwen/Qwen2-1.5B-Instruct (Fast, Small)",
    '2': "Qwen/Qwen2-7B-Instruct (Balanced)",
    '3': "meta-llama/Meta-Llama-3-8B-Instruct (Strong)",
    '5': "microsoft/phi-2 (Small, Good for Simpler Tasks)",
    '6': "google/gemma-2-9b-it (Google's Gemma 2 Instruct)"
}
model_id_map = {
    '1': "Qwen/Qwen2-1.5B-Instruct",
    '2': "Qwen/Qwen2-7B-Instruct",
    '3': "meta-llama/Meta-Llama-3-8B-Instruct",
    '5': "microsoft/phi-2",
    '6': "google/gemma-2-9b-it"
}

print("Please select the LLM model (using a number):")
for key, name in model_choices_display.items():
    print(f" ({key}) {name}")
receive_model_key = input("Select here: ")
while receive_model_key not in model_id_map:
    print("Invalid selection. Please choose from the list.")
    receive_model_key = input("Select here: ")
model_id = model_id_map[receive_model_key]
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

# Load model
model = None
if tokenizer:
    try:
        if device.type == "mps": dtype = torch.float16
        elif device.type == "cuda": dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        else: dtype = torch.float32
        
        print(f"Loading model {model_id} with dtype: {dtype}")
        model_load_args = {"torch_dtype": dtype, "cache_dir": "."}
        if model_id == "microsoft/phi-2": # Phi-2 often needs trust_remote_code
             model_load_args["trust_remote_code"] = True

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_load_args
        ).to(device)
        model.eval() 
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")

if tokenizer and tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
        print("Tokenizer does not have a pad_token_id. Setting it to eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        print("Error: Tokenizer has no pad_token_id and no eos_token_id. Manual intervention needed.")
        # Fallback: tokenizer.add_special_tokens({'pad_token': '[PAD]'}) and resize model embeddings if necessary
        # This is more complex and model-specific. For now, we'll proceed hoping eos_token_id exists.


# --- LLM Interaction Functions ---
def get_llm_response(prompt_text):
    if model is None or tokenizer is None:
        print("Model or tokenizer not loaded. Cannot get response.")
        return "ERROR_MODEL_NOT_LOADED"
    if tokenizer.pad_token_id is None: # Guard against pad_token_id being None before generation
        print("Error: pad_token_id is None. Cannot generate response.")
        return "ERROR_PAD_TOKEN_NONE"


    # Truncate prompt if it's too long from the beginning (keeping the end)
    # This is a simple truncation. More sophisticated methods might be needed for very long histories.
    prompt_max_len = tokenizer.model_max_length - 50 # Reserve 50 tokens for generation
    if len(tokenizer.encode(prompt_text)) > prompt_max_len:
        print(f"Warning: Prompt too long, truncating. Original length: {len(tokenizer.encode(prompt_text))}")
        # A simple way to truncate while keeping the end of the prompt (history and current choice)
        # This might cut off important initial context or examples if history is very long.
        # A better approach might be to summarize or truncate the middle of the history.
        lines = prompt_text.split('\n')
        if len(lines) > 20: # Keep critical parts: initial instructions, examples, and recent history
            # Keep first ~5 lines (instructions, examples part 1)
            # Keep last ~10 lines (current situation, history, choice line)
            # This is a heuristic.
            preserved_header_lines = 5 
            preserved_tail_lines = 15 # Should capture "Current situation", some history, and "Your choice"
            
            if len(lines) > preserved_header_lines + preserved_tail_lines:
                 prompt_text = "\n".join(lines[:preserved_header_lines]) + \
                               "\n... [History Truncated] ...\n" + \
                               "\n".join(lines[-(preserved_tail_lines):])
            
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=False).to(device) # Truncation now handled manually above
    input_length = inputs.input_ids.shape[1]
    
    if input_length >= tokenizer.model_max_length -10 : # If still too long after simple truncation
        print(f"Warning: Input length {input_length} is close to model_max_length {tokenizer.model_max_length}. Response might be cut.")
        # Forcefully truncate input_ids if absolutely necessary
        inputs.input_ids = inputs.input_ids[:, -(tokenizer.model_max_length - 50):]
        inputs.attention_mask = inputs.attention_mask[:, -(tokenizer.model_max_length - 50):]
        input_length = inputs.input_ids.shape[1]


    # print(f"\n--- Sending Prompt (length {input_length}) ---\n{prompt_text}\n----------------------------")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=10, 
            do_sample=True, 
            temperature=0.6,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id 
        )
    
    newly_generated_tokens = outputs[0, input_length:]
    generated_text = tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip()
    # print(f"Raw LLM Output: '{generated_text}'")
    return generated_text

def get_llm_choice_from_response(raw_response, valid_options):
    # print(f"Attempting to parse: '{raw_response}' from options: {valid_options}")
    
    # Create a regex pattern that looks for any of the valid options as whole words
    # It should be case-insensitive and allow for optional surrounding non-alphanumeric characters or start/end of string
    # Escape special characters in options, then join with '|'
    patterns = [re.escape(option) for option in valid_options]
    regex_pattern = r'(?:^|\s|[^\w])(' + r'|'.join(patterns) + r')(?:$|\s|[^\w])'

    match = re.search(regex_pattern, raw_response, re.IGNORECASE)
    if match:
        # print(f"Regex match found: {match.group(1)}")
        # Find which of the valid_options it matched (case-insensitively)
        for option in valid_options:
            if option.lower() == match.group(1).lower():
                return option
    
    # Fallback: simpler check for just the option string if regex fails (e.g. if response is *only* the option)
    cleaned_response_lines = [line.strip() for line in raw_response.split('\n') if line.strip()]
    if cleaned_response_lines:
        first_potential_choice = cleaned_response_lines[0].split(' ')[0].strip() # Take first word of first non-empty line
        for option in valid_options:
            if option.lower() == first_potential_choice.lower():
                # print(f"Fallback simple match: {option}")
                return option

    print(f"Warning: LLM response '{raw_response}' did not clearly match any valid option: {valid_options}.")
    return None

# --- Simulation Function ---
def simulate_scenario_outcome(chosen_option_id, current_option_probabilities, outcome_scores):
    if chosen_option_id not in current_option_probabilities:
        print(f"Error: Option {chosen_option_id} not found in current probabilities for simulation.")
        # Find a default error outcome if possible or just a generic one
        if "ErrorOutcome" in outcome_scores: return "ErrorOutcome", outcome_scores["ErrorOutcome"]
        return "SIMULATION_ERROR", -100 

    probabilities_for_chosen_option = current_option_probabilities[chosen_option_id]
    possible_outcomes = list(probabilities_for_chosen_option.keys())
    outcome_probs_values = list(probabilities_for_chosen_option.values())

    # Ensure probabilities sum to 1 and are non-negative
    sum_probs = sum(outcome_probs_values)
    if not np.isclose(sum_probs, 1.0) or np.any(np.array(outcome_probs_values) < 0):
        print(f"Warning/Error: Probabilities for {chosen_option_id} are invalid (sum: {sum_probs}, values: {outcome_probs_values}). Attempting to normalize or defaulting.")
        if sum_probs > 0 and not np.any(np.array(outcome_probs_values) < 0): # Can normalize if sum > 0 and no negatives
            outcome_probs_values = np.array(outcome_probs_values) / sum_probs
            if not np.isclose(sum(outcome_probs_values), 1.0): # Check after normalization
                 print(f"Error: Probabilities for {chosen_option_id} still invalid after normalization. Defaulting outcome.")
                 if possible_outcomes: chosen_outcome_name = possible_outcomes[0] # Fallback
                 else: return "CRITICAL_SIM_ERROR", -200
            else:
                 chosen_outcome_name = np.random.choice(possible_outcomes, p=outcome_probs_values)
        else: # Cannot normalize (e.g. sum is zero, or negative probabilities)
            print(f"Error: Cannot normalize probabilities for {chosen_option_id}. Defaulting outcome.")
            if possible_outcomes: chosen_outcome_name = possible_outcomes[0] # Fallback
            else: return "CRITICAL_SIM_ERROR", -200
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
            ev += prob * outcome_scores.get(outcome_name, 0)
        evs[option] = ev
    return evs

# --- Main Test Loop Function ---
def run_scenario_episode(scenario_key, num_iterations=10):
    if model is None or tokenizer is None or tokenizer.pad_token_id is None:
        print("Model, tokenizer or pad_token_id not properly loaded/set. Skipping episode.")
        return {"avg_score": 0, "optimal_choice_rate": 0, "valid_response_rate": 0, "total_score":0}

    print(f"\n--- Starting Episode for Scenario: {ALL_SCENARIOS[scenario_key]['scenario_name']} ---")
    
    current_scenario_config = copy.deepcopy(ALL_SCENARIOS[scenario_key])
    # Since no narrative updates, active_probabilities will be initial_probabilities for the whole episode
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
    optimal_option_id_ev = max(expected_values, key=expected_values.get) if expected_values else None
    print(f"Expected Values for this scenario's options: {expected_values}")
    print(f"Optimal option based on EV for this scenario: {optimal_option_id_ev} (EV: {expected_values.get(optimal_option_id_ev, 'N/A')})")


    for i in range(num_iterations):
        iteration_num = i + 1
        # print(f"--- {scenario_key} - Iteration {iteration_num}/{num_iterations} ---") # Less verbose

        # Construct the prompt
        # No narrative update section as per user request
        prompt = f"""You are an AI agent. {task_description}
{option_intro}
Based on the history of choices and their outcomes, decide which option to pick next.
Output ONLY the option identifier ({', '.join(valid_options)}). Do not include any other words, explanations, or formatting.

{examples_for_prompt}

Current situation:
History:
{previous_outputs_history}Your choice ({', '.join(valid_options)}):"""

        ai_response_raw = get_llm_response(prompt)
        llm_choice = get_llm_choice_from_response(ai_response_raw, valid_options)

        history_item_prefix = "Turn" # Default
        if scenario_key == "StockMarket": history_item_prefix = f"Q{iteration_num}"
        elif scenario_key == "CropYield": history_item_prefix = f"Season {iteration_num}"
        elif scenario_key == "PrizeDoor": history_item_prefix = f"Round {iteration_num}"
        elif scenario_key == "SmugglingRoutes": history_item_prefix = f"Trip {iteration_num}"
        elif scenario_key == "DietaryAdvisor": history_item_prefix = f"Client {chr(ord('A') + i)}"
        elif scenario_key == "SlotMachine": history_item_prefix = f"Turn {iteration_num}"


        if llm_choice:
            valid_responses_count += 1
            # print(f"LLM chose: {llm_choice}") # Less verbose

            if optimal_option_id_ev and llm_choice == optimal_option_id_ev:
                optimal_choices_made_count += 1
            
            outcome_name, outcome_score = simulate_scenario_outcome(
                llm_choice,
                active_probabilities, 
                outcome_scores
            )
            total_score_this_episode += outcome_score
            
            previous_outputs_history += f"{history_item_prefix}: Chose {llm_choice}, Result: {outcome_name}\n"
            # print(f"Outcome: {outcome_name} (Score: {outcome_score})") # Less verbose
        else:
            # print(f"LLM response '{ai_response_raw}' was invalid. Assigning penalty.") # Less verbose
            total_score_this_episode -= 20 
            previous_outputs_history += f"{history_item_prefix}: Chose Invalid, Result: Penalty Applied\n"


        # current_avg_score = total_score_this_episode / iteration_num
        # current_optimal_rate = optimal_choices_made_count / iteration_num if iteration_num > 0 else 0
        # print(f"Iter {iteration_num}: Avg Score: {current_avg_score:.2f}, Optimal Rate: {current_optimal_rate:.2f}")


    avg_score = total_score_this_episode / num_iterations if num_iterations > 0 else 0
    optimal_choice_rate = optimal_choices_made_count / num_iterations if num_iterations > 0 else 0
    valid_response_rate = valid_responses_count / num_iterations if num_iterations > 0 else 0
    
    print(f"--- Episode End for {scenario_key} ---")
    print(f"  Total Score: {total_score_this_episode}, Avg Score: {avg_score:.2f}")
    print(f"  Optimal Choices: {optimal_choices_made_count}/{num_iterations} (Rate: {optimal_choice_rate:.2f})")
    print(f"  Valid Responses: {valid_responses_count}/{num_iterations} (Rate: {valid_response_rate:.2f})")
    print("-----------------------------------------")
    
    return {
        "avg_score": avg_score, 
        "optimal_choice_rate": optimal_choice_rate,
        "valid_response_rate": valid_response_rate,
        "total_score": total_score_this_episode
    }

# --- Main Execution ---
if __name__ == "__main__":
    if model is None or tokenizer is None or (tokenizer and tokenizer.pad_token_id is None):
        print("Exiting: Model, Tokenizer, or pad_token_id failed to load/set properly.")
        exit()

    num_episodes_per_scenario = input("Enter number of episodes per scenario (e.g., 3): ")
    try:
        num_episodes_per_scenario = int(num_episodes_per_scenario)
        if num_episodes_per_scenario <= 0: raise ValueError
    except ValueError:
        print("Invalid input. Defaulting to 2 episodes.")
        num_episodes_per_scenario = 2

    iterations_per_episode = input("Enter number of iterations (turns) per episode (e.g., 10): ")
    try:
        iterations_per_episode = int(iterations_per_episode)
        if iterations_per_episode <= 0: raise ValueError
    except ValueError:
        print("Invalid input. Defaulting to 7 iterations.")
        iterations_per_episode = 7


    overall_results = {}
    
    print("\nAvailable Scenarios:")
    for i, key in enumerate(ALL_SCENARIOS.keys()):
        print(f" ({i+1}) {key}")
    print(" (A) All Scenarios")
    
    scenario_selection_input = input("Choose scenario number or 'A' for All: ").strip().upper()

    scenarios_to_run = []
    if scenario_selection_input == 'A':
        scenarios_to_run = list(ALL_SCENARIOS.keys())
    else:
        try:
            choice_idx = int(scenario_selection_input) - 1
            if 0 <= choice_idx < len(ALL_SCENARIOS):
                scenarios_to_run.append(list(ALL_SCENARIOS.keys())[choice_idx])
            else:
                print("Invalid scenario number. Exiting.")
                exit()
        except ValueError:
            print("Invalid input. Exiting.")
            exit()

    for scenario_id in scenarios_to_run:
        print(f"\n\n================== TESTING SCENARIO: {ALL_SCENARIOS[scenario_id]['scenario_name']} ==================")
        scenario_episode_total_scores = []
        scenario_optimal_rates = []
        scenario_valid_response_rates = []

        for i in range(num_episodes_per_scenario):
            print(f"--- Running Episode {i+1}/{num_episodes_per_scenario} for {scenario_id} ---")
            episode_stats = run_scenario_episode(scenario_id, iterations_per_episode)
            scenario_episode_total_scores.append(episode_stats["total_score"]) # Store total score for overall average
            scenario_optimal_rates.append(episode_stats["optimal_choice_rate"])
            scenario_valid_response_rates.append(episode_stats["valid_response_rate"])
        
        overall_results[scenario_id] = {
            "mean_total_score_per_episode": np.mean(scenario_episode_total_scores) if scenario_episode_total_scores else 0,
            "std_total_score_per_episode": np.std(scenario_episode_total_scores) if scenario_episode_total_scores else 0,
            "mean_optimal_choice_rate": np.mean(scenario_optimal_rates) if scenario_optimal_rates else 0,
            "mean_valid_response_rate": np.mean(scenario_valid_response_rates) if scenario_valid_response_rates else 0,
        }
        print(f"================== COMPLETED SCENARIO: {ALL_SCENARIOS[scenario_id]['scenario_name']} ==================")
        print(f"  Mean Total Score per Episode: {overall_results[scenario_id]['mean_total_score_per_episode']:.2f} (Std: {overall_results[scenario_id]['std_total_score_per_episode']:.2f})")
        print(f"  Mean Optimal Choice Rate: {overall_results[scenario_id]['mean_optimal_choice_rate']:.2f}")
        print(f"  Mean Valid Response Rate: {overall_results[scenario_id]['mean_valid_response_rate']:.2f}")


    print("\n\n=============== OVERALL SUMMARY ACROSS SCENARIOS (using selected LLM) ===============")
    print(f"LLM Used: {model_id}")
    print(f"Episodes per Scenario: {num_episodes_per_scenario}, Iterations per Episode: {iterations_per_episode}")
    for scenario_id, results in overall_results.items():
        print(f"\nScenario: {ALL_SCENARIOS[scenario_id]['scenario_name']}")
        print(f"  Mean Total Score per Episode: {results['mean_total_score_per_episode']:.2f} (Std: {results['std_total_score_per_episode']:.2f})")
        print(f"  Mean Optimal Choice Rate (picked EV-best based on initial probabilities): {results['mean_optimal_choice_rate']:.2f}")
        print(f"  Mean Valid Response Rate: {results['mean_valid_response_rate']:.2f}")
