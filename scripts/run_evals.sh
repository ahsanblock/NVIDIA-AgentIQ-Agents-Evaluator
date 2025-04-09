#!/bin/bash

set -e

# Check for NVIDIA API key
if [ -z "${NVIDIA_API_KEY}" ]; then
    echo "NVIDIA_API_KEY is not set. Please provide your NVIDIA API key:"
    read -s NVIDIA_API_KEY
    echo "Exporting NVIDIA_API_KEY..."
fi

# Export the API key
export NVIDIA_API_KEY

# Default values
MODELS=()
DATASET=""
EXAMPLE_CONFIGS=true
CLEAR_RESULTS=false
RESULTS_DIR="results"
# Add delay parameter with default value of 10 seconds
DELAY_BETWEEN_MODELS=10

# Validate NVIDIA model name format
validate_model_name() {
    local model="$1"
    # Check if model name follows pattern: org/model-name
    if [[ ! "$model" =~ ^[a-zA-Z0-9-]+/[a-zA-Z0-9-]+$ ]] && [[ ! "$model" =~ ^[a-zA-Z0-9-]+/[a-zA-Z0-9-]+/[a-zA-Z0-9-]+$ ]]; then
        echo "Error: Invalid model name format: $model"
        echo "Model name should be in format: organization/model-name or organization/model/variant"
        return 1
    fi
    return 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      if ! validate_model_name "$2"; then
        exit 1
      fi
      MODELS+=("$2")
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --example-configs)
      EXAMPLE_CONFIGS=true
      shift
      ;;
    --no-example-configs)
      EXAMPLE_CONFIGS=false
      shift
      ;;
    --clear-results)
      CLEAR_RESULTS=true
      shift
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --delay)
      DELAY_BETWEEN_MODELS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set default models if none specified
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(
    "meta-llama/Llama-2-8b-chat-hf"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "microsoft/phi-3-mini"
    "microsoft/phi-3-medium"
    "meta-llama/Llama-2-70b-chat-hf"
  )
fi

# Validate all models
for MODEL in "${MODELS[@]}"; do
  if ! validate_model_name "$MODEL"; then
    exit 1
  fi
done

# Change the hardcoded path to use environment variable
# Find AgentIQ installation
if [ -n "${AGENTIQ_PATH}" ]; then
  EXTERNAL_AGENTIQ_PATH="${AGENTIQ_PATH}"
else
  # Default locations to check
  EXTERNAL_AGENTIQ_PATH="../AgentIQ"
  if [ ! -d "$EXTERNAL_AGENTIQ_PATH" ]; then
    EXTERNAL_AGENTIQ_PATH="/opt/AgentIQ"
  fi
fi

if command -v aiq &> /dev/null; then
  AIQ_CMD="aiq"
elif [ -d "$EXTERNAL_AGENTIQ_PATH" ]; then
  echo "AgentIQ command not found in PATH, but detected at $EXTERNAL_AGENTIQ_PATH"
  echo "Activating external AgentIQ environment..."
  source "$EXTERNAL_AGENTIQ_PATH/.venv/bin/activate"
  AIQ_CMD="$EXTERNAL_AGENTIQ_PATH/.venv/bin/aiq"
else
  echo "AgentIQ not found. Please install it or activate its environment."
  echo "Set the AGENTIQ_PATH environment variable to point to your AgentIQ installation."
  echo "Example: export AGENTIQ_PATH=/path/to/AgentIQ"
  exit 1
fi

# Path to AgentIQ examples
AIQ_EXAMPLES_DIR="${EXTERNAL_AGENTIQ_PATH}/examples"
AIQ_CONFIG_DIR="${AIQ_EXAMPLES_DIR}/email_phishing_analyzer/configs"

# Local configs dir
LOCAL_CONFIG_DIR="configs"

# Create the AgentIQ configs directory if it doesn't exist
mkdir -p "$AIQ_CONFIG_DIR"

# Copy template file to AgentIQ configs
TEMPLATE_FILE="${LOCAL_CONFIG_DIR}/template.yml"
if [ -f "$TEMPLATE_FILE" ]; then
  echo "Copying template config to AgentIQ configs directory..."
  cp "$TEMPLATE_FILE" "${AIQ_CONFIG_DIR}/template.yml"
fi

# Clear previous results if requested
if [ "$CLEAR_RESULTS" = true ]; then
  echo "Clearing previous results in ${RESULTS_DIR}..."
  mkdir -p "$RESULTS_DIR"
  find "$RESULTS_DIR" -mindepth 1 -not -name ".gitkeep" -exec rm -rf {} \; 2>/dev/null || true
  
  echo "Clearing previous plots..."
  mkdir -p plots
  find plots -mindepth 1 -not -name ".gitkeep" -exec rm -rf {} \; 2>/dev/null || true
fi

echo
echo "========== LLM EVALUATION: Email Phishing Detection =========="
echo "Models: ${MODELS[*]}"
echo "Dataset: ${DATASET:-data/test_data.csv}"
echo "Output directory: ${RESULTS_DIR}"
echo "Using example configs: ${EXAMPLE_CONFIGS}"
echo "Clear previous results: ${CLEAR_RESULTS}"
echo "Delay between models: ${DELAY_BETWEEN_MODELS} seconds"
echo "=========================================================="
echo

# Set the dataset path
if [ -n "$DATASET" ]; then
  DATASET_PATH="$(realpath "$DATASET")"
else
  DATASET_PATH="$(realpath data/test_data.csv)"
fi

echo "Using dataset at: ${DATASET_PATH}"
echo

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Run evaluations for each model
FIRST_MODEL=true
for MODEL in "${MODELS[@]}"; do
  # Add delay between model evaluations, but not before the first one
  if [ "$FIRST_MODEL" = true ]; then
    FIRST_MODEL=false
  else
    echo "Waiting ${DELAY_BETWEEN_MODELS} seconds to avoid rate limiting..."
    sleep "$DELAY_BETWEEN_MODELS"
  fi

  echo "---- Starting evaluation for model: ${MODEL} ----"
  
  # Create a template-based config for this model
  TEMPLATE_FILE="${AIQ_CONFIG_DIR}/template.yml"
  
  # Create a sanitized model name for file paths
  SANITIZED_MODEL=$(echo "$MODEL" | tr '/' '_')
  CONFIG_FILE="${AIQ_CONFIG_DIR}/config-${SANITIZED_MODEL}.yml"
  
  # Create a simplified config file containing just the model name
  echo "Creating minimal model config from template..."
  
  # Extract a simpler model ID from the full model name for the output directory
  MODEL_ID=$(echo "$MODEL" | sed 's|.*/||')
  
  cat > "$CONFIG_FILE" << EOF
llms:
  nim_llm:
    _type: nim
    model_name: $MODEL
    temperature: 0.0
    max_tokens: 512
EOF
  
  # Create output directory for this model
  MODEL_OUTPUT_DIR="${RESULTS_DIR}/${MODEL_ID}"
  mkdir -p "$MODEL_OUTPUT_DIR"
  MODEL_OUTPUT_DIR="$(realpath "${MODEL_OUTPUT_DIR}")"
  
  # Create a temporary config file with updated paths
  TEMP_CONFIG_FILE=$(mktemp)
  echo "Creating temporary config file with updated dataset path and output directory..."
  
  # Use Python script to update config file
  python3 create_temp_config.py "$CONFIG_FILE" "${TEMP_CONFIG_FILE}" "${NVIDIA_API_KEY}" "${DATASET_PATH}" "${MODEL_OUTPUT_DIR}"
  if [ $? -ne 0 ]; then
    echo "Failed to update config file with Python script."
    exit 1
  fi
  
  # Create a temporary file to store the exit code
  EXIT_CODE_FILE=$(mktemp)
  
  # Run the evaluation
  (
    # Show which API key is being used
    echo "Using NVIDIA API key starting with: ${NVIDIA_API_KEY:0:5}..."
    
    # Run the evaluation with the AIQ command directly
    $AIQ_CMD eval --config_file "${TEMP_CONFIG_FILE}"
      
    echo $? > "$EXIT_CODE_FILE"
    
    # Rename workflow output to results.json if exists
    if [ -f "${MODEL_OUTPUT_DIR}/workflow_output.json" ]; then
      cp "${MODEL_OUTPUT_DIR}/workflow_output.json" "${MODEL_OUTPUT_DIR}/results.json"
    fi
    
    # Generate metrics.json
    echo "Generating metrics.json from results..."
    if [ -f "${MODEL_OUTPUT_DIR}/results.json" ]; then
      # Extract metrics using Python
      python3 -c "
import json
import sys
import statistics
from datetime import datetime

def analyze_response(generated_answer, label):
    # Convert everything to lowercase for easier comparison
    answer = generated_answer.lower()
    label = label.lower()
    
    # Keywords that indicate phishing detection
    phish_indicators = ['phishing', 'scam', 'suspicious', 'fraud', 'fake']
    
    is_phishing = label == 'phish'
    detected_phishing = any(indicator in answer for indicator in phish_indicators)
    
    # Return True if the model correctly identified phishing/benign
    return detected_phishing == is_phishing

try:
    with open('${MODEL_OUTPUT_DIR}/results.json', 'r') as f:
        results = json.load(f)
    
    correct = 0
    total = 0
    latencies = []
    
    for item in results:
        if 'label' in item and 'generated_answer' in item:
            total += 1
            if analyze_response(item['generated_answer'], item['label']):
                correct += 1
            
            # Extract latency from intermediate steps
            steps = item.get('intermediate_steps', [])
            for step in steps:
                payload = step.get('payload', {})
                if 'event_timestamp' in payload and 'span_event_timestamp' in payload:
                    latency = payload['event_timestamp'] - payload['span_event_timestamp']
                    if latency > 0:  # Only add valid latencies
                        latencies.append(latency)
    
    accuracy = correct / total if total > 0 else 0
    avg_latency = statistics.mean(latencies) if latencies else 0
    
    metrics = {
        'model_id': '${MODEL_ID}',
        'model_name': '${MODEL}',
        'accuracy': accuracy,
        'average_latency': avg_latency,
        'correct_predictions': correct,
        'total_examples': total,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('${MODEL_OUTPUT_DIR}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f'Created metrics.json with accuracy: {accuracy*100:.1f}%, latency: {avg_latency:.2f}s ({correct}/{total} correct)')
    
except Exception as e:
    print(f'Error generating metrics: {e}')
    sys.exit(1)
      "
      
      if [ $? -eq 0 ]; then
        echo "Successfully generated metrics.json"
      else
        echo "Failed to generate metrics.json"
      fi
    else
      echo "No results file found, skipping metrics generation"
    fi
    
    # Clean up temporary config file
    rm "${TEMP_CONFIG_FILE}"
    
  ) &
  
  # Wait for the evaluation to complete
  wait
  
  # Get the exit code
  EXIT_CODE=$(cat "$EXIT_CODE_FILE")
  rm "$EXIT_CODE_FILE"
  
  if [ $EXIT_CODE -ne 0 ]; then
    echo "Evaluation for ${MODEL} failed with exit code ${EXIT_CODE}"
  else
    echo "---- Completed evaluation for ${MODEL} ----"
  fi
  
  echo "Results saved to: ${RESULTS_DIR}/${MODEL_ID}/results.json"
  echo "Profiling metrics saved to: ${RESULTS_DIR}/${MODEL_ID}/workflow_profiling_metrics.json"
  echo
done

echo "All evaluations completed successfully!"
echo "Results saved to: ${RESULTS_DIR}"
echo "To visualize the results, run: python3 scripts/generate_visualizations.py --results_dir ${RESULTS_DIR}"