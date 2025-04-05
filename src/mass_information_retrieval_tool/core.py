import os
import yaml
import json
import logging
import pandas as pd
from google import genai # Correct import for google-genai package
from google.genai.types import GenerationConfig # Import GenerationConfig directly
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Tuple
from joblib import Parallel, delayed # Import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Type mapping for schema validation
TYPE_MAPPING = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads and validates the configuration file."""
    logging.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        raise

    # Basic validation
    required_keys = ['input_csv', 'output_csv', 'key_column', 'gemini_model', 'api_key_env_var', 'schema', 'prompt_template']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in configuration: {key}")

    # Validate schema type
    if not isinstance(config['schema'], dict):
         raise ValueError("Configuration 'schema' must be a dictionary.")

    # Validate and set default for max_parallel_requests
    if 'max_parallel_requests' not in config:
        config['max_parallel_requests'] = 5 # Default value
        logging.info("max_parallel_requests not found in config, defaulting to 5.")
    elif not isinstance(config['max_parallel_requests'], int) or config['max_parallel_requests'] <= 0:
        raise ValueError("Configuration 'max_parallel_requests' must be a positive integer.")

    logging.info("Configuration loaded successfully.")
    return config

# Add genai.Client to import if it's not already implicitly covered by 'from google import genai'
# It seems it is, so no import change needed here.

def configure_gemini(api_key_env_var: str) -> genai.Client: # Add return type hint
    """Loads API key and configures the Gemini client."""
    load_dotenv() # Load .env file if present
    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise ValueError(f"API key environment variable '{api_key_env_var}' not found. Ensure it's set in your environment or a .env file.")
    # Instantiate and return the client
    client = genai.Client(api_key=api_key)
    logging.info("Gemini client configured.")
    return client # Return the client

def generate_schema_json(schema: Dict[str, str]) -> str:
    """Creates a JSON string representation of the schema for the prompt."""
    schema_example = {}
    for key, type_str in schema.items():
        if type_str == "int":
            schema_example[key] = 0
        elif type_str == "float":
            schema_example[key] = 0.0
        elif type_str == "bool":
            schema_example[key] = False
        else: # default to string
            schema_example[key] = "string_value"
    return json.dumps(schema_example, indent=2)

# Modify to accept the client object and use client.models.generate_content
def call_gemini_api(client: genai.Client, model_name: str, prompt: str) -> Optional[Dict[str, Any]]: # Add client param
    """Calls the Gemini API using the provided client and attempts to parse the JSON response."""
    try:
        # Call generate_content via client.models, passing the model name
        # Define the search tool
        search_tool = {"google_search": {}}
        response = client.models.generate_content( # Use client.models
            model=model_name, # Pass model name as parameter
            contents=[prompt], # Pass prompt within 'contents' list
            config={ # Pass config as a dictionary directly
                'tools': [search_tool] # Enable Google Search tool
            }
        )
        # Extract JSON text - handle potential variations in response structure
        json_text = response.text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]

        return json.loads(json_text)
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to decode JSON response: {e}\nResponse text: {response.text[:500]}...") # Log first 500 chars
        return None
    except Exception as e:
        # Catch other potential API errors (rate limits, connection issues, etc.)
        # Use logging.exception to include the full stack trace
        logging.exception("Gemini API call failed:")
        # Consider adding retry logic here if needed
        return None


def validate_and_convert(data: Any, target_type_str: str) -> Any:
    """Validates and converts data to the target type."""
    if data is None:
        return None # Keep nulls as they are (will become NaN in pandas)

    target_type = TYPE_MAPPING.get(target_type_str)
    if not target_type:
        logging.warning(f"Unsupported type '{target_type_str}' in schema. Treating as string.")
        return str(data)

    try:
        # Handle boolean conversion specifically if needed (e.g., "true"/"false" strings)
        if target_type == bool:
            if isinstance(data, str):
                if data.lower() == 'true':
                    return True
                elif data.lower() == 'false':
                    return False
                else:
                    # Attempt numeric conversion for bool (0=False, non-zero=True)
                    try:
                        return bool(int(data))
                    except ValueError:
                         raise ValueError(f"Cannot convert string '{data}' to boolean")
            else:
                 return bool(data) # Standard Python bool conversion
        else:
            return target_type(data)
    except (ValueError, TypeError) as e:
        logging.warning(f"Type conversion failed for value '{data}' to type '{target_type_str}': {e}")
        return None # Return None on conversion failure

# --- Helper function for parallel processing ---
def process_row_with_gemini(
    index: int,
    item: Any,
    client: genai.Client,
    model_name: str,
    prompt_template: str,
    schema_json_str: str,
    schema: Dict[str, str],
    key_column: str
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Processes a single row: calls Gemini API and validates the result."""
    if pd.isna(item):
        logging.warning(f"Skipping row {index + 1} due to missing value in key column '{key_column}'.")
        return index, None # Return index and None data

    logging.info(f"Processing row {index + 1}: Item = '{item}'")
    prompt = prompt_template.format(item=item, schema_json=schema_json_str)

    # Call Gemini API using the client
    retrieved_data = call_gemini_api(client, model_name, prompt)

    processed_row_data = {}
    if retrieved_data:
        logging.debug(f"Raw data received for '{item}': {retrieved_data}")
        # Populate dictionary with validated data for this row
        for col_name, type_str in schema.items():
            value = retrieved_data.get(col_name) # Use .get() for safe access
            validated_value = validate_and_convert(value, type_str)
            processed_row_data[col_name] = validated_value
        return index, processed_row_data
    else:
        logging.warning(f"No valid data retrieved from Gemini for item: '{item}'")
        # Return index and None data if API call failed or returned invalid data
        return index, None


def process_csv(config: Dict[str, Any]) -> None:
    """Main function to process the CSV file using parallel requests."""
    # 1. Configure Gemini and get the client
    try:
        client = configure_gemini(config['api_key_env_var']) # Assign returned client
    except ValueError as e:
        logging.error(f"Failed to configure Gemini: {e}") # Update log message
        return # Exit if client setup fails

    # 2. Load paths and settings from config
    input_path = config['input_csv']
    output_path = config['output_csv']
    key_column = config['key_column']
    schema = config['schema']
    model_name = config['gemini_model']
    prompt_template = config['prompt_template']
    max_parallel_requests = config['max_parallel_requests'] # Get parallel config

    try:
        df = pd.read_csv(input_path)
        logging.info(f"Read {len(df)} rows from {input_path}")
    except FileNotFoundError:
        logging.error(f"Input CSV file not found: {input_path}")
        return
    except Exception as e:
        logging.error(f"Error reading CSV file {input_path}: {e}")
        return

    if key_column not in df.columns:
        logging.error(f"Key column '{key_column}' not found in {input_path}")
        return

    # Prepare schema JSON for the prompt
    schema_json_str = generate_schema_json(schema)

    # Add new columns for the schema fields, initialized with None (becomes NaN)
    for col_name in schema.keys():
        if col_name not in df.columns:
            df[col_name] = None
        else:
             logging.warning(f"Schema column '{col_name}' already exists in the input CSV. Its values might be overwritten.")

    # --- Parallel Processing using joblib ---
    logging.info(f"Starting parallel processing with {max_parallel_requests} workers.")

    # Prepare inputs for each job
    job_inputs = [
        (index, row[key_column]) for index, row in df.iterrows()
    ]

    # Run jobs in parallel
    results = Parallel(n_jobs=max_parallel_requests, backend="threading")(
        delayed(process_row_with_gemini)(
            index, item, client, model_name, prompt_template, schema_json_str, schema, key_column
        ) for index, item in job_inputs
    )

    # --- Update DataFrame with results ---
    logging.info("Updating DataFrame with processed results.")
    update_count = 0
    for index, processed_data in results:
        if processed_data: # Check if data was successfully retrieved and processed
            for col_name, value in processed_data.items():
                 # Use loc for reliable setting by index
                df.loc[index, col_name] = value
            update_count += 1
        # Rows with None data (due to NaN input or API failure) keep their initial None/NaN values

    logging.info(f"Successfully updated {update_count} rows out of {len(df)}.")

    # --- Save the enriched DataFrame ---
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir: # Check if output_dir is not empty (i.e., not just a filename)
             os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_path, index=False)
        logging.info(f"Enriched data saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error saving output CSV file {output_path}: {e}")
