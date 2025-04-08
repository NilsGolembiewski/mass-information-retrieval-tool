import os
import yaml
import json
import logging
import pandas as pd
from google import genai  # Correct import for google-genai package

# from google.genai.types import GenerationConfig # No longer explicitly needed for basic calls
from googleapiclient.discovery import (
    build as build_google_service,
)  # Renamed to avoid conflict
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Tuple  # Added List
from joblib import Parallel, delayed  # Import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

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
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        raise

    # --- Configuration Validation ---
    logging.info("Validating configuration...")
    # Define required keys for the new workflow
    required_keys = [
        "input_csv",
        "output_csv",
        "key_columns",
        "gemini_model",
        "api_key_env_var",
        "schema",
        "query_generation_prompt_template",
        "final_processing_prompt_template",
        "google_search_api_key_env_var",
        "google_cse_id_env_var",
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(
            f"Missing required keys in configuration: {', '.join(missing_keys)}"
        )

    # Validate key_columns type
    if not isinstance(config["key_columns"], list) or not all(
        isinstance(col, str) for col in config["key_columns"]
    ):
        raise ValueError("Configuration 'key_columns' must be a list of strings.")
    if not config["key_columns"]:
        raise ValueError("Configuration 'key_columns' cannot be empty.")

    # Validate schema type
    if not isinstance(config["schema"], dict):
        raise ValueError("Configuration 'schema' must be a dictionary.")

    # Validate and set default for max_parallel_requests
    if "max_parallel_requests" not in config:
        config["max_parallel_requests"] = 5  # Default value
        logging.info("max_parallel_requests not found in config, defaulting to 5.")
    elif (
        not isinstance(config["max_parallel_requests"], int)
        or config["max_parallel_requests"] <= 0
    ):
        raise ValueError(
            "Configuration 'max_parallel_requests' must be a positive integer."
        )

    # Validate and set default for google_search_num_results
    if "google_search_num_results" not in config:
        config["google_search_num_results"] = 2  # Default value
        logging.info("google_search_num_results not found in config, defaulting to 2.")
    elif (
        not isinstance(config["google_search_num_results"], int)
        or config["google_search_num_results"] <= 0
    ):
        raise ValueError(
            "Configuration 'google_search_num_results' must be a positive integer."
        )

    # Validate prompt templates are strings
    if not isinstance(config["query_generation_prompt_template"], str):
        raise ValueError(
            "Configuration 'query_generation_prompt_template' must be a string."
        )
    if not isinstance(config["final_processing_prompt_template"], str):
        raise ValueError(
            "Configuration 'final_processing_prompt_template' must be a string."
        )

    logging.info("Configuration loaded and validated successfully.")
    return config


# Add genai.Client to import if it's not already implicitly covered by 'from google import genai'
# It seems it is, so no import change needed here.


def configure_gemini(api_key_env_var: str) -> genai.Client:  # Add return type hint
    """Loads API key and configures the Gemini client."""
    load_dotenv()  # Load .env file if present
    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise ValueError(
            f"API key environment variable '{api_key_env_var}' not found. Ensure it's set in your environment or a .env file."
        )
    # Instantiate and return the client
    client = genai.Client(api_key=api_key)
    logging.info("Gemini client configured.")
    return client  # Return the client


# --- New Function: Configure Google Custom Search ---
def configure_google_search(api_key_env_var: str, cse_id_env_var: str) -> Optional[Any]:
    """Loads API key and CSE ID and configures the Google Custom Search service."""
    load_dotenv()  # Ensure .env is loaded
    api_key = os.getenv(api_key_env_var)
    cse_id = os.getenv(cse_id_env_var)

    if not api_key:
        logging.error(
            f"Google Search API key environment variable '{api_key_env_var}' not found."
        )
        return None  # Indicate failure
    if not cse_id:
        logging.error(
            f"Google Search CSE ID environment variable '{cse_id_env_var}' not found."
        )
        return None  # Indicate failure

    try:
        service = build_google_service("customsearch", "v1", developerKey=api_key)
        logging.info("Google Custom Search service configured.")
        return service
    except Exception as e:
        logging.error(
            f"Failed to build Google Custom Search service: {e}", exc_info=True
        )
        return None


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
        else:  # default to string
            schema_example[key] = "string_value"
    return json.dumps(schema_example, indent=2)


# --- New Function: Call Gemini for Query Generation ---
def call_gemini_for_query(
    client: genai.Client, model_name: str, prompt: str
) -> Optional[str]:
    """Calls the Gemini API to generate a search query string (no tools enabled)."""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
            # No 'config' or 'tools' needed here
        )
        query = response.text.strip()
        logging.debug(f"Generated query: {query}")
        return query
    except Exception:
        logging.exception(
            f"Gemini API call for query generation failed (prompt: {prompt[:100]}...):"
        )
        return None


# --- New Function: Execute Google Search ---
def execute_google_search(
    service: Any, cse_id: str, query: str, num_results: int
) -> Optional[str]:
    """Executes a single Google Custom Search query and returns concatenated snippets."""
    if not service:
        logging.error("Google Search service not configured, skipping search.")
        return None
    if not query:
        logging.warning("Empty query provided, skipping search.")
        return None

    try:
        logging.debug(f"Executing Google Search for query: {query}")
        results = service.cse().list(q=query, cx=cse_id, num=num_results).execute()

        items = results.get("items", [])
        if not items:
            logging.debug(f"No Google Search results found for query: {query}")
            return None  # Or return an empty string "" if preferred

        # Concatenate snippets or titles/links if snippets are missing
        snippets = [item.get("snippet", item.get("title", "")) for item in items]
        result_text = "\n---\n".join(snippets).strip()
        logging.debug(f"Search results text (first 100 chars): {result_text[:100]}...")
        return result_text

    except HttpError as e:
        logging.error(
            f"Google Search API HTTP error for query '{query}': {e}", exc_info=True
        )
        return None
    except Exception as e:
        logging.error(
            f"Unexpected error during Google Search for query '{query}': {e}",
            exc_info=True,
        )
        return None


# --- New Function: Execute All Google Searches for a Row ---
def execute_all_google_searches(
    queries: Dict[str, Optional[str]],
    search_service: Any,
    cse_id: str,
    num_results: int,
) -> Dict[str, Optional[str]]:
    """Executes searches for all generated queries for a row."""
    search_results = {}
    for field_name, query in queries.items():
        if query:  # Only search if a query was successfully generated
            result_text = execute_google_search(
                search_service, cse_id, query, num_results
            )
            search_results[field_name] = result_text
        else:
            search_results[field_name] = None  # Mark as None if query generation failed
            logging.warning(
                f"Skipping search for field '{field_name}' due to missing query."
            )
    return search_results


# --- New Function: Generate Search Queries for a Row ---
def generate_search_queries_for_row(
    client: genai.Client,
    model_name: str,
    key_values: Dict[str, Any],
    schema: Dict[str, str],
    query_template: str,
) -> Dict[str, Optional[str]]:
    """Generates a search query for each field in the schema using Gemini."""
    queries = {}
    key_values_json_str = json.dumps(key_values)
    logging.debug(f"Generating search queries for key values: {key_values_json_str}")

    for field_name in schema.keys():
        prompt = query_template.format(
            key_values_json=key_values_json_str, field_name=field_name
        )
        query = call_gemini_for_query(client, model_name, prompt)
        queries[field_name] = query  # Store query (or None if generation failed)

    return queries


# --- New Function: Call Gemini for Final Processing ---
def call_gemini_for_processing(
    client: genai.Client, model_name: str, prompt: str
) -> Optional[Dict[str, Any]]:
    """Calls Gemini API to process aggregated results into final JSON (no tools)."""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
            # No 'config' or 'tools' needed here
        )
        # Extract JSON text - handle potential variations in response structure
        json_text = response.text.strip()
        logging.debug(f"Raw JSON response from final processing: {json_text[:500]}...")
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]

        return json.loads(json_text)
    except json.JSONDecodeError as e:
        logging.warning(
            f"Failed to decode final JSON response: {e}\nResponse text: {response.text[:500]}..."
        )
        return None
    except Exception:
        logging.exception("Gemini API call for final processing failed:")
        return None


def validate_and_convert(data: Any, target_type_str: str) -> Any:
    """Validates and converts data to the target type."""
    if data is None:
        return None  # Keep nulls as they are (will become NaN in pandas)

    target_type = TYPE_MAPPING.get(target_type_str)
    if not target_type:
        logging.warning(
            f"Unsupported type '{target_type_str}' in schema. Treating as string."
        )
        return str(data)

    try:
        # Handle boolean conversion specifically if needed (e.g., "true"/"false" strings)
        if target_type is bool:
            if isinstance(data, str):
                if data.lower() == "true":
                    return True
                elif data.lower() == "false":
                    return False
                else:
                    # Attempt numeric conversion for bool (0=False, non-zero=True)
                    try:
                        return bool(int(data))
                    except ValueError:
                        raise ValueError(f"Cannot convert string '{data}' to boolean")
            else:
                return bool(data)  # Standard Python bool conversion
        else:
            return target_type(data)
    except (ValueError, TypeError) as e:
        logging.warning(
            f"Type conversion failed for value '{data}' to type '{target_type_str}': {e}"
        )
        return None  # Return None on conversion failure


# --- Refactored Helper function for parallel processing ---
def process_row_multi_step(
    index: int,
    key_values: Dict[str, Any],
    gemini_client: genai.Client,
    # Removed search_service parameter
    config: Dict[str, Any],  # Pass the whole config for convenience
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """
    Processes a single row using the multi-step workflow.
    Initializes its own Google Search service client for thread safety.
    """
    key_values_json_str = json.dumps(key_values)
    logging.info(
        f"--- Processing row {index + 1}: Key Values = {key_values_json_str} ---"
    )

    # --- Step 0: Configure Google Search Service (Thread-Local) ---
    # Each thread configures its own service instance
    search_service = configure_google_search(
        config["google_search_api_key_env_var"], config["google_cse_id_env_var"]
    )
    if not search_service:
        logging.error(
            f"Row {index + 1}: Failed to configure Google Search service in thread. Skipping searches."
        )
        # Decide how to handle this - maybe return None early?
        # For now, proceed, but searches will fail/be skipped in execute_all_google_searches

    # --- Step 1: Generate Search Queries ---
    queries = generate_search_queries_for_row(
        gemini_client,
        config["gemini_model"],
        key_values,
        config["schema"],
        config["query_generation_prompt_template"],
    )
    logging.debug(f"Row {index + 1}: Generated queries: {queries}")

    # --- Step 2: Execute Google Searches ---
    search_results = execute_all_google_searches(
        queries,
        search_service,
        os.getenv(config["google_cse_id_env_var"]),  # Get CSE ID from env
        config["google_search_num_results"],
    )
    logging.debug(f"Row {index + 1}: Search results obtained.")

    # --- Step 3: Aggregate Search Results for Final Prompt ---
    aggregate_results_text = ""
    for field_name, result_text in search_results.items():
        query = queries.get(field_name, "N/A")  # Get the query used
        if result_text:
            aggregate_results_text += (
                f"Results for '{field_name}' (query: '{query}'):\n{result_text}\n\n"
            )
        else:
            aggregate_results_text += (
                f"No results found for '{field_name}' (query: '{query}').\n\n"
            )
    aggregate_results_text = aggregate_results_text.strip()

    # --- Step 4: Call Gemini for Final Processing ---
    schema_json_str = generate_schema_json(config["schema"])
    final_prompt = config["final_processing_prompt_template"].format(
        key_values_json=key_values_json_str,
        schema_json=schema_json_str,
        search_results_aggregate=aggregate_results_text,
    )

    final_data = call_gemini_for_processing(
        gemini_client, config["gemini_model"], final_prompt
    )

    # --- Step 5: Validate and Return ---
    processed_row_data = {}
    if final_data and isinstance(final_data, dict):
        logging.debug(f"Row {index + 1}: Raw final data received: {final_data}")
        for col_name, type_str in config["schema"].items():
            value = final_data.get(col_name)
            validated_value = validate_and_convert(value, type_str)
            processed_row_data[col_name] = validated_value
        logging.info(f"Row {index + 1}: Successfully processed.")
        return index, processed_row_data
    else:
        logging.warning(f"Row {index + 1}: Failed to retrieve or process final data.")
        return index, None


def process_csv(config: Dict[str, Any]) -> None:
    """Main function to process the CSV file using the multi-step parallel workflow."""
    # 1. Configure Gemini Client
    try:
        gemini_client = configure_gemini(config["api_key_env_var"])
    except ValueError as e:
        logging.error(f"Failed to configure Gemini: {e}")
        return

    # 2. Load paths and settings from config (Search service configured per thread now)
    input_path = config["input_csv"]
    output_path = config["output_csv"]
    key_columns = config["key_columns"]
    schema = config["schema"]  # Keep schema for validation/column creation
    max_parallel_requests = config["max_parallel_requests"]

    # 4. Load DataFrame
    try:
        df = pd.read_csv(input_path)
        logging.info(f"Read {len(df)} rows from '{input_path}'")
    except FileNotFoundError:
        logging.error(f"Input CSV file not found: '{input_path}'")
        return
    except Exception as e:
        logging.error(f"Error reading CSV file '{input_path}': {e}")
        return

    # 5. Validate Key Columns
    missing_cols = [col for col in key_columns if col not in df.columns]
    if missing_cols:
        logging.error(
            f"Key columns not found in '{input_path}': {', '.join(missing_cols)}"
        )
        return

    # 6. Prepare DataFrame: Add output columns if they don't exist
    for col_name in schema.keys():
        if col_name not in df.columns:
            df[col_name] = pd.NA  # Use pandas NA for better type handling
        else:
            logging.warning(
                f"Schema column '{col_name}' already exists in the input CSV. Its values might be overwritten."
            )

    # 7. Prepare Inputs for Parallel Processing
    job_inputs = []
    for index, row in df.iterrows():
        key_values = {col: row[col] for col in key_columns}
        # Handle potential NaN/None in key values before passing to API
        cleaned_key_values = {k: v for k, v in key_values.items() if pd.notna(v)}
        if len(cleaned_key_values) < len(key_values):
            logging.warning(
                f"Row {index + 1} has missing value(s) in key columns: {key_values}. Using available keys: {cleaned_key_values}"
            )
        if not cleaned_key_values:
            logging.error(
                f"Row {index + 1} has NO valid key values. Skipping processing for this row."
            )
            job_inputs.append((index, None))  # Mark row for skipping
        else:
            job_inputs.append((index, cleaned_key_values))

    # 8. Run Parallel Processing
    logging.info(
        f"Starting parallel processing with {max_parallel_requests} workers..."
    )
    results = Parallel(n_jobs=max_parallel_requests, backend="threading")(
        delayed(process_row_multi_step)(
            index,
            key_vals,
            gemini_client,
            # Removed search_service from arguments passed to delayed function
            config,  # Pass the whole config dictionary
        )
        for index, key_vals in job_inputs
        if key_vals is not None  # Skip rows marked for skipping
    )

    # 9. Update DataFrame with Results
    logging.info("Updating DataFrame with processed results...")
    update_count = 0
    processed_indices = set()  # Keep track of processed rows
    for result in results:  # Iterate through actual results
        if result:  # Check if the result tuple itself is not None
            index, processed_data = result
            processed_indices.add(index)
            if processed_data:  # Check if data was successfully retrieved and processed
                for col_name, value in processed_data.items():
                    # Use loc for reliable setting by index
                    df.loc[index, col_name] = value
                update_count += 1
            # else: Row processing failed, keep original NA/None

    total_processed = len(processed_indices)
    total_skipped = len(df) - total_processed
    logging.info(
        f"Processing finished. Successfully retrieved data for {update_count} rows out of {total_processed} processed rows ({total_skipped} rows skipped due to missing keys)."
    )

    # 10. Save Enriched DataFrame
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Enriched data saved to: '{output_path}'")
    except Exception as e:
        logging.error(f"Error saving output CSV file '{output_path}': {e}")
