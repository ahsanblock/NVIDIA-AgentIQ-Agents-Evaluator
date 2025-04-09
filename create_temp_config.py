#!/usr/bin/env python3
import sys
import yaml
import os
import copy

# Standard prompt that will be used for all configurations
STANDARD_PROMPT = """Analyze the following email for phishing signals.

Email content:
{body}

Is this a phishing email? Provide a brief explanation."""

# Model name mapping from user-friendly names to NVIDIA AI Endpoints names
# This allows users to use simple names like "llama-3.1-8b" while ensuring
# we use the correct fully-qualified model name for the API
MODEL_MAPPING = {
    # Common LLaMA models
    "llama-3.1-8b": "meta/llama-3.1-8b-instruct",
    "llama-3.1-8b-instruct": "meta/llama-3.1-8b-instruct",
    "llama-3.1-70b": "meta/llama-3.1-70b-instruct",
    "llama-3.1-70b-instruct": "meta/llama-3.1-70b-instruct",
    "llama-3.3-8b": "meta/llama-3.3-8b-instruct",
    "llama-3.3-8b-instruct": "meta/llama-3.3-8b-instruct",
    "llama-3.3-70b": "meta/llama-3.3-70b-instruct",
    "llama-3.3-70b-instruct": "meta/llama-3.3-70b-instruct",
    
    # Mistral models
    "mistral-7b": "mistralai/mistral-7b-instruct-v0.2",
    "mistral-7b-instruct": "mistralai/mistral-7b-instruct-v0.2",
    "mistral-7b-instruct-v0.2": "mistralai/mistral-7b-instruct-v0.2",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct-v0.1",
    "mixtral-8x7b-instruct": "mistralai/mixtral-8x7b-instruct-v0.1",
    "mixtral-8x7b-instruct-v0.1": "mistralai/mixtral-8x7b-instruct-v0.1",
    "mixtral-8x22b": "mistralai/mixtral-8x22b-instruct-v0.1",
    "mixtral-8x22b-instruct": "mistralai/mixtral-8x22b-instruct-v0.1",
    "mixtral-8x22b-instruct-v0.1": "mistralai/mixtral-8x22b-instruct-v0.1",
    
    # Microsoft models
    "phi-3-mini": "microsoft/phi-3-mini-4k-instruct",
    "phi-3-mini-4k": "microsoft/phi-3-mini-4k-instruct",
    "phi-3-mini-4k-instruct": "microsoft/phi-3-mini-4k-instruct",
    "phi-3-medium": "microsoft/phi-3-medium-4k-instruct",
    "phi-3-medium-4k": "microsoft/phi-3-medium-4k-instruct",
    "phi-3-medium-4k-instruct": "microsoft/phi-3-medium-4k-instruct",
    
    # Google models
    "gemma-7b": "google/gemma-7b-it",
    "gemma-7b-it": "google/gemma-7b-it",
}

def get_full_model_name(model_name):
    """
    Get the full model name from the mapping, or return the original name if not found.
    This allows for both the short friendly names and the full vendor/model names to work.
    
    Args:
        model_name (str): The model name to look up
        
    Returns:
        str: The full model name with vendor prefix
    """
    # If already has a vendor prefix (containing /), assume it's a full name
    if '/' in model_name:
        return model_name
        
    # Otherwise look up in mapping
    return MODEL_MAPPING.get(model_name, model_name)

def get_simple_model_name(full_model_name):
    """
    Extract a simple model name from a full model name for use in filenames and directories.
    
    Args:
        full_model_name (str): The full model name like "meta/llama-3.1-8b-instruct"
        
    Returns:
        str: A simplified name like "llama-3.1-8b-instruct"
    """
    # If it contains a slash, take the part after the slash
    if '/' in full_model_name:
        return full_model_name.split('/', 1)[1]
    return full_model_name

def create_temp_config(input_file, output_file, api_key, dataset_path, output_dir):
    """
    Create a temporary config file with standardized settings.
    
    Args:
        input_file (str): Path to the input config file
        output_file (str): Path to save the updated config
        api_key (str): NVIDIA API key
        dataset_path (str): Path to the dataset
        output_dir (str): Directory to save output
    """
    # Load the template config
    template_path = "configs/template.yml"
    
    try:
        # First load the input config
        with open(input_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Extract model name from the config
        model_name = None
        if 'llms' in config and 'nim_llm' in config['llms'] and 'model_name' in config['llms']['nim_llm']:
            model_name = config['llms']['nim_llm']['model_name']
        
        # If we don't have a model name but have a full name, extract it (for backward compatibility)
        if not model_name and input_file.endswith('.yml'):
            # Try to extract from filename like "config-meta/llama-3.1-8b-instruct.yml"
            filename = os.path.basename(input_file)
            if filename.startswith('config-'):
                extracted_name = filename[len('config-'):-4]  # Remove 'config-' prefix and '.yml' suffix
                model_name = extracted_name
            
        if model_name:
            # Ensure we have the full model name with vendor prefix
            full_model_name = get_full_model_name(model_name)
            simple_model_name = get_simple_model_name(full_model_name)
            
            print(f"Using model: {full_model_name} (simplified as '{simple_model_name}')")
            model_name = full_model_name
        
        # Now load the template for standardization
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                template = yaml.safe_load(f)
                
            # Merge template with config to prioritize standardized settings
            standardized_config = copy.deepcopy(template)
            
            # Keep the original model name
            if model_name:
                if 'llms' not in standardized_config:
                    standardized_config['llms'] = {}
                
                # Set up the model names for all LLM configurations
                for llm_key in ['nim_llm', 'nim_rag_eval_llm', 'nim_trajectory_eval_llm']:
                    if llm_key not in standardized_config['llms']:
                        standardized_config['llms'][llm_key] = {}
                        
                    standardized_config['llms'][llm_key]['model_name'] = model_name
                    
                    # Make sure the _type is set correctly
                    standardized_config['llms'][llm_key]['_type'] = 'nim'
                    
                    # Set standard max_tokens
                    standardized_config['llms'][llm_key]['max_tokens'] = 512
                    
                    # Set appropriate temperatures
                    if llm_key == 'nim_rag_eval_llm':
                        standardized_config['llms'][llm_key]['temperature'] = 0.9
                    else:
                        standardized_config['llms'][llm_key]['temperature'] = 0.0
                
                # Fix evaluator LLM references
                if 'evaluators' in standardized_config.get('eval', {}):
                    for evaluator_name, evaluator in standardized_config['eval']['evaluators'].items():
                        if 'trajectory' in evaluator_name:
                            evaluator['llm_name'] = 'nim_trajectory_eval_llm'
                        elif 'rag' in evaluator_name:
                            evaluator['llm_name'] = 'nim_rag_eval_llm'
            
            # Use the standardized config instead of the original
            config = standardized_config
        
    except Exception as e:
        print(f"Error loading config from {input_file} or template: {e}")
        return False
    
    # Standardize the prompt
    if 'functions' in config and 'email_phishing_analyzer' in config['functions']:
        config['functions']['email_phishing_analyzer']['prompt'] = STANDARD_PROMPT
    
    # Update paths
    if 'eval' in config and 'general' in config['eval']:
        config['eval']['general']['output_dir'] = output_dir
        
        # Also update any MODEL_ID placeholders in the output directory
        if isinstance(output_dir, str) and output_dir.endswith('/'):
            model_folder = os.path.basename(os.path.dirname(output_dir))
            if 'MODEL_ID' in config['eval']['general']['output_dir']:
                config['eval']['general']['output_dir'] = config['eval']['general']['output_dir'].replace('MODEL_ID', model_folder)
        
    if 'eval' in config and 'general' in config['eval'] and 'dataset' in config['eval']['general']:
        config['eval']['general']['dataset']['file_path'] = dataset_path
    
    # Ensure the workflow field exists and remove front_end field if it exists
    if 'front_end' in config:
        del config['front_end']
        
    if 'workflow' not in config:
        print(f"Adding required 'workflow' field to config")
        config['workflow'] = {
            "_type": "tool_calling_agent",
            "tool_names": ["email_phishing_analyzer"],
            "llm_name": "nim_llm",
            "verbose": True,
            "retry_parsing_errors": True,
            "max_retries": 1
        }
    elif 'llm' in config['workflow'] and 'tool_names' not in config['workflow']:
        # Convert older workflow format to new format
        model_name = None
        if 'model_name' in config['workflow']['llm']:
            model_name = config['workflow']['llm']['model_name']
            
        config['workflow'] = {
            "_type": "tool_calling_agent", 
            "tool_names": ["email_phishing_analyzer"],
            "llm_name": "nim_llm",
            "verbose": True,
            "retry_parsing_errors": True,
            "max_retries": 1
        }
        
        # If we had a model name, make sure it's preserved in the llms section
        if model_name and 'llms' in config and 'nim_llm' in config['llms']:
            config['llms']['nim_llm']['model_name'] = model_name
    
    # Update API key
    if 'llms' in config:
        for llm_name in config['llms']:
            if 'credentials' not in config['llms'][llm_name]:
                config['llms'][llm_name]['credentials'] = {}
            config['llms'][llm_name]['credentials']['api_key'] = api_key
    
    # Save updated config
    try:
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        print(f"Error saving config to {output_file}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python create_temp_config.py <input_file> <output_file> <api_key> <dataset_path> <output_dir>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    api_key = sys.argv[3]
    dataset_path = sys.argv[4] 
    output_dir = sys.argv[5]
    
    if create_temp_config(input_file, output_file, api_key, dataset_path, output_dir):
        print(f"Config updated and saved to {output_file}")
        sys.exit(0)
    else:
        sys.exit(1) 