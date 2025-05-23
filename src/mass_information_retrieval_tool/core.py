import os
import yaml
import json
import logging
import pandas as pd
from google import genai  # Correct import for google-genai package
import html
from bs4 import BeautifulSoup
import csv
import io

# from google.genai.types import GenerationConfig # No longer explicitly needed
# Removed googleapiclient imports
from dotenv import load_dotenv
from typing import (
    Dict,
    Any,
    Optional,
    Tuple,
)  # Removed List import as it's not used directly
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

def csv_join(row: list[str]) -> str:
    """
    Joins a list of strings into a single string, escaping commas and quotes.
    """
    # Escape commas and quotes in each field
    result = io.StringIO()

    writer = csv.writer(result, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(row)
    return result.getvalue().strip()


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
    # Define required keys for the field-by-field workflow
    required_keys = [
        "input_csv",
        "output_csv",
        "key_columns",
        "gemini_model",
        "api_key_env_var",
        "schema",
        "single_field_prompt_template",  # Added new required prompt
    ]
    # Removed keys related to external Google Search and previous multi-step prompts
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
    # Remove validation for google_search_num_results as it's no longer used

    # Validate the new single_field_prompt_template
    if not isinstance(config["single_field_prompt_template"], str):
        raise ValueError(
            "Configuration 'single_field_prompt_template' must be a string."
        )
    # Remove validation for old/removed prompt templates

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


# Removed configure_google_search function


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


# Removed functions:
# - call_gemini_for_query
# - execute_google_search
# - execute_all_google_searches
# - generate_search_queries_for_row
# - call_gemini_for_processing


def extract_sources_from_candidate(candidate):
    """
    Extracts source URLs from a Gemini candidate's grounding_metadata using BeautifulSoup.
    Returns a list of URLs (may be empty).
    """
    urls = []
    grounding_metadata = getattr(candidate, 'grounding_metadata', None)
    if not grounding_metadata:
        return urls

    # 1. Extract from rendered_content (HTML)
    search_entry_point = getattr(grounding_metadata, 'search_entry_point', None)
    if search_entry_point and hasattr(search_entry_point, 'rendered_content'):
        html_content = search_entry_point.rendered_content
        soup = BeautifulSoup(html_content, 'html.parser')
        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            urls.append(link)

    # 2. Extract from grounding_supports
    grounding_supports = getattr(grounding_metadata, 'grounding_supports', None)
    if grounding_supports:
        for support in grounding_supports:
            # Try both object and dict access
            uri = None
            if hasattr(support, 'web') and hasattr(support.web, 'uri'):
                uri = getattr(support.web, 'uri', None)
            elif isinstance(support, dict) and 'uri' in support:
                uri = support['uri']
            if uri:
                urls.append(uri)

    # 3. Extract from retrieval_metadata.sources
    retrieval_metadata = getattr(grounding_metadata, 'retrieval_metadata', None)
    if retrieval_metadata and hasattr(retrieval_metadata, 'sources'):
        for source in retrieval_metadata.sources:
            uri = getattr(source, 'uri', None)
            if uri:
                urls.append(uri)

    return urls


# --- New Function: Call Gemini for a Single Field ---
def call_gemini_for_field(
    client: genai.Client, model_name: str, prompt: str
) -> Optional[tuple]:
    """
    Calls the Gemini API for a single field, enabling the Google Search tool,
    and expects a JSON response like {"value": ...}.
    Returns a tuple: (extracted value, source URL(s)).
    """
    search_tool = {"google_search": {}}  # Correctly instantiate the search tool
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
            config={
                "tools": [search_tool]
            },
        )

        # --- Log Token Usage ---
        try:
            usage = response.usage_metadata
            logging.info(
                f"Gemini Token Usage - Prompt: {usage.prompt_token_count}, Candidates: {usage.candidates_token_count}, Total: {usage.total_token_count}"
            )
        except Exception as e:
            logging.warning(f"Could not retrieve token usage metadata: {e}")
        # --- End Log Token Usage ---

        # Extract JSON text - handle potential variations in response structure
        json_text = response.text.strip()
        logging.debug(f"Raw JSON response for field: {json_text[:500]}...")

        # Look for a ```json block in the response
        if "```json" in json_text:
            start_index = json_text.find("```json") + 7
            end_index = json_text.rfind("```")
            if end_index > start_index:
                json_text = json_text[start_index:end_index].strip()
            else:
                logging.warning(
                    "Closing ``` not found or invalid for JSON block. Using raw response."
                )
        else:
            logging.warning("No ```json block found. Using raw response.")

        # Parse the JSON and extract the 'value'
        parsed_json = json.loads(json_text)
        value = None
        if isinstance(parsed_json, dict) and "value" in parsed_json:
            value = parsed_json["value"]
        else:
            logging.warning(f"Unexpected JSON structure received: {json_text}")
            value = None

        # --- Extract source(s) using grounding_metadata strategy ---
        source_urls = None
        try:
            candidate = None
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
            if candidate:
                urls = extract_sources_from_candidate(candidate)
                if urls:
                    source_urls = csv_join(urls)
        except Exception as e:
            logging.warning(f"Could not extract citations/source URLs: {e}")
            source_urls = None

        return value, source_urls

    except json.JSONDecodeError as e:
        logging.warning(
            f"Failed to decode JSON response for field: {e}\nResponse text: {response.text[:500]}..."
        )
        return None, None
    except Exception:
        logging.exception("Gemini API call for field failed:")
        return None, None


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
                    except ValueError as e:
                        raise ValueError(f"Cannot convert string '{data}' to boolean") from e
            else:
                return bool(data)  # Standard Python bool conversion
        else:
            return target_type(data)
    except (ValueError, TypeError) as e:
        logging.warning(
            f"Type conversion failed for value '{data}' to type '{target_type_str}': {e}"
        )
        return None  # Return None on conversion failure


# --- New Helper function for parallel processing (Field-by-Field) ---
def process_row_field_by_field(
    index: int,
    key_values: Dict[str, Any],
    gemini_client: genai.Client,
    config: Dict[str, Any],
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """
    Processes a single row by querying Gemini for each field individually,
    with Gemini's search tool enabled.
    Now also extracts and stores the source (citation URL) for each field.
    """
    key_values_json_str = json.dumps(key_values)
    logging.info(
        f"--- Processing row {index + 1}: Key Values = {key_values_json_str} ---"
    )

    processed_row_data = {}
    schema = config["schema"]
    model_name = config["gemini_model"]
    prompt_template = config["single_field_prompt_template"]

    # Iterate through each field defined in the schema
    for field_name, field_type in schema.items():
        logging.debug(f"Row {index + 1}: Querying for field '{field_name}'...")

        # Format the prompt for the current field
        prompt = prompt_template.format(
            key_values_json=key_values_json_str,
            field_name=field_name,
            field_type=field_type,  # Pass type info to prompt if needed
        )

        # Call Gemini API for this specific field (with search enabled)
        retrieved_value, source_url = call_gemini_for_field(gemini_client, model_name, prompt)

        # Validate and store the result for this field
        if retrieved_value is not None:
            validated_value = validate_and_convert(retrieved_value, field_type)
            processed_row_data[field_name] = validated_value
            logging.debug(
                f"Row {index + 1}: Field '{field_name}' value: {validated_value}"
            )
        else:
            processed_row_data[field_name] = (
                None  # Store None if retrieval/parsing failed
            )
            logging.warning(
                f"Row {index + 1}: Failed to retrieve or parse value for field '{field_name}'."
            )
        # Store the source URL(s) in a new column
        processed_row_data[f"{field_name}__source"] = source_url

    # Return the aggregated data for the row
    logging.info(f"Row {index + 1}: Finished processing all fields.")
    return index, processed_row_data


def process_csv(config: Dict[str, Any]) -> None:
    """Main function to process the CSV file using the field-by-field parallel workflow."""
    # 1. Configure Gemini Client
    try:
        gemini_client = configure_gemini(config["api_key_env_var"])
    except ValueError as e:
        logging.error(f"Failed to configure Gemini: {e}")
        return

    # 2. Load paths and settings from config (No separate search service config needed)
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
        # Add the source column for each field
        source_col = f"{col_name}__source"
        if source_col not in df.columns:
            df[source_col] = pd.NA

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
        delayed(process_row_field_by_field)(  # Call the new processing function
            index,
            key_vals,
            gemini_client,
            config,
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
