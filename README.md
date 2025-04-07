# Mass Information Retrieval Tool

A Python tool that retrieves structured information for a list of items from a CSV file using a multi-step process involving Google Gemini and Google Custom Search, based on a YAML configuration file.

### Workflow Overview

1.  **Query Generation:** For each item in the input CSV and each field defined in the target schema, the tool asks Google Gemini to generate an optimal Google search query.
2.  **Search Execution:** The generated queries are executed using the Google Custom Search API.
3.  **Result Processing:** The aggregated search results are then fed back into Google Gemini, along with the original item details and the target schema, to generate the final structured JSON output.

This multi-step approach aims to leverage Gemini's query generation capabilities and Google Search's real-time information access, followed by Gemini's ability to synthesize and structure the gathered data.

## Prerequisites

*   Python (version specified in `pyproject.toml`, e.g., ^3.9)
*   [Poetry](https://python-poetry.org/) for dependency management.

## Setup

### For Usage
There are multiple options for installing this tool:

-  Install using `pipx`: `pipx install git+https://github.com/NilsGolembiewski/mass-information-retrieval-tool`
-  Install using `pip`: `pip install git+https://github.com/NilsGolembiewski/mass-information-retrieval-tool`

If you are planning to only use this tool and not modify it, you can continue at [Configuration](#configuration-configyaml).
  
### For Development

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository-url>
    cd mass-information-retrieval-tool
    ```

2.  **Install dependencies:**
    ```bash
    poetry install
    ```

3.  **Set up API Keys:**
    *   Create a file named `.env` in the project root directory (`/home/nils/Projects/mass-information-retrieval-tool/.env`).
    *   Add your **Google Gemini API key** and your **Google Custom Search API key** and **Custom Search Engine (CSE) ID** to the `.env` file. The key names should match those specified in your `config.yaml` (see defaults below):
        ```dotenv
        GEMINI_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY
        GOOGLE_CSE_API_KEY=YOUR_ACTUAL_GOOGLE_CSE_API_KEY
        GOOGLE_CSE_ID=YOUR_ACTUAL_GOOGLE_CSE_ID
        ```
    *   You can obtain a Google Custom Search API key and create a CSE ID via the [Google Cloud Console](https://console.cloud.google.com/) and the [Programmable Search Engine control panel](https://programmablesearchengine.google.com/).
    *   Ensure `.env` is listed in your `.gitignore` file to prevent committing secrets.

## Configuration (`config.yaml`)

The tool's behavior is controlled by a `config.yaml` file (default location is the project root). Create this file if it doesn't exist and configure the following options:

*   `input_csv`: (Required) Path to the input CSV file containing the items to look up.
*   `output_csv`: (Required) Path where the enriched output CSV file will be saved. The directory will be created if it doesn't exist.
*   `key_columns`: (Required) A list of column names in the `input_csv` that together identify the items (e.g., `["Company Name", "Product ID"]`) for which information should be retrieved.
*   `gemini_model`: (Required) The specific Google Gemini model to use (e.g., `gemini-1.5-flash`).
*   `api_key_env_var`: (Required) The name of the environment variable holding your Google Gemini API key.
*   `google_search_api_key_env_var`: (Required) The name of the environment variable holding your Google Custom Search API key.
*   `google_cse_id_env_var`: (Required) The name of the environment variable holding your Google Custom Search Engine ID.
*   `google_search_num_results`: (Optional) The number of search results to fetch for each generated query (default: 2).
*   `schema`: (Required) A dictionary defining the desired output structure.
    *   Keys: The names of the new columns to be added to the output CSV.
    *   Values: The expected data type for each column (`str`, `int`, `float`, `bool`). The tool will attempt to convert the final data received from Gemini to these types.
*   `query_generation_prompt_template`: (Required) A template string for the prompt sent to Gemini to generate a search query for a *single field*.
    *   Use `{key_values_json}` as a placeholder for a JSON object containing the key-value pairs from the `key_columns` for the current row.
    *   Use `{field_name}` as a placeholder for the specific schema field the query is being generated for.
*   `final_processing_prompt_template`: (Required) A template string for the prompt sent to Gemini to process the aggregated search results.
    *   Use `{key_values_json}` as a placeholder for the original key-value pairs.
    *   Use `{schema_json}` as a placeholder for a JSON representation of the desired output structure.
    *   Use `{search_results_aggregate}` as a placeholder for the combined text of all search results gathered for the row.
*   `prompt_template`: (Optional, Legacy) The original prompt template used in the previous single-step workflow. Kept for reference but not used by the current multi-step process.

**Example `config.yaml`:**

```yaml
# --- Configuration for Mass Information Retrieval Tool ---

# Input CSV file path
input_csv: "data/companies.csv" # IMPORTANT: Update this path

# Output CSV file path (will be created)
output_csv: "data/companies_enriched.csv" # IMPORTANT: Update this path

# List of column names in the input CSV containing the key information
key_columns: ["Company Name"] # IMPORTANT: Update this list

# Google Gemini API configuration
gemini_model: "gemini-2.0-flash" # Or another suitable model
api_key_env_var: "GEMINI_API_KEY"

# Parallel processing configuration
max_parallel_requests: 5 # Adjust as needed

# Google Custom Search API configuration
google_search_api_key_env_var: "GOOGLE_CSE_API_KEY"
google_cse_id_env_var: "GOOGLE_CSE_ID"
google_search_num_results: 2

# Schema definition for the information to retrieve
schema:
  approx_employees: "int"
  industry: "str"
  headquarters_location: "str"
  founded_year: int
  website: str

# Prompt template for generating a Google Search query for a specific field
query_generation_prompt_template: |
  Based on the following item details:
  {key_values_json}

  Generate the single best Google search query string to find the value for the field '{field_name}'.
  Output ONLY the raw query string, without any explanation or formatting.

# Prompt template for processing aggregated search results into the final JSON
final_processing_prompt_template: |
  Context:
  - Original item details: {key_values_json}
  - Target JSON schema structure: {schema_json}
  - Collected Google Search results for various fields:
  {search_results_aggregate}

  Task:
  Analyze the provided context and search results. Generate a single JSON object that strictly adheres to the target schema structure. Populate the JSON object fields using the most relevant information found in the search results.
  - If a value for a field can be accurately determined from the search results, include it.
  - If a value cannot be determined or is ambiguous, use `null` for that field.
  - Ensure the output is only the valid JSON object, enclosed in ```json ... ``` if necessary, but preferably just the raw JSON.
```

## Usage

Ensure your API keys are set in the `.env` file (see [Setup](#setup)).

Run the tool using the script entry point provided by Poetry, loading the environment variables from `.env` using `dotenv run --`:

```bash
dotenv run -- poetry run mass-retrieve-info --config config.yaml
```

*   Replace `config.yaml` with the actual path to your configuration file if it's different or located elsewhere.
*   The script will execute the multi-step workflow: generate queries, perform searches, process results, and save the enriched data to the `output_csv`.
*   Progress and any errors will be logged to the console.

## Example Usage (Using `example_data`)

This repository includes an `example_data` directory containing sample files to demonstrate the tool's functionality.

1.  **Configuration (`example_data/config.yaml`):**
    This file is pre-configured to use the example input/output files and defines a schema for company information:
    ```yaml
    # --- Configuration for Mass Information Retrieval Tool ---

    # Input CSV file path
    input_csv: "example_data/input.csv"

    # Output CSV file path (will be created)
    output_csv: "example_data/output.csv"

    # List of column names in the input CSV containing the key information
    key_columns: ["CompanyName"]

    # Google Gemini API configuration
    gemini_model: "gemini-2.0-flash"
    api_key_env_var: "GEMINI_API_KEY"

    # Parallel processing configuration
    max_parallel_requests: 5

    # Google Custom Search API configuration
    google_search_api_key_env_var: "GOOGLE_CSE_API_KEY"
    google_cse_id_env_var: "GOOGLE_CSE_ID"
    google_search_num_results: 2

    # Schema definition for the information to retrieve
    schema:
      number_of_employees: int
      annual_revenue: float
      headquarters_location: str
      year_founded: int
      industry: str
      website: str

    # Prompt template for generating a Google Search query for a specific field
    query_generation_prompt_template: |
      Based on the following item details:
      {key_values_json}

      Generate the single best Google search query string to find the value for the field '{field_name}'.
      Output ONLY the raw query string, without any explanation or formatting.

    # Prompt template for processing aggregated search results into the final JSON
    final_processing_prompt_template: |
      Context:
      - Original item details: {key_values_json}
      - Target JSON schema structure: {schema_json}
      - Collected Google Search results for various fields:
      {search_results_aggregate}

      Task:
      Analyze the provided context and search results. Generate a single JSON object that strictly adheres to the target schema structure. Populate the JSON object fields using the most relevant information found in the search results.
      - If a value for a field can be accurately determined from the search results, include it.
      - If a value cannot be determined or is ambiguous, use `null` for that field.
      - Ensure the output is only the valid JSON object, enclosed in ```json ... ``` if necessary, but preferably just the raw JSON.
    ```

2.  **Input Data (`example_data/input.csv`):**
    The input file contains a list of company names:
    ```csv
    CompanyName,Region,ContactPerson
    "Apple Inc.","North America",""
    "Microsoft Corporation","North America",""
    "Alphabet Inc.","North America",""
    "Amazon.com, Inc.","North America",""
    "Meta Platforms, Inc.","North America",""
    "Cisco Systems, Inc.","North America",""
    ```

3.  **Run the Example:**
    Execute the tool using the example configuration:
    ```bash
    dotenv run -- poetry run mass-retrieve-info --config example_data/config.yaml
    ```
    *(Ensure your `GEMINI_API_KEY`, `GOOGLE_CSE_API_KEY`, and `GOOGLE_CSE_ID` are set in the `.env` file as described in the Setup section).*

4.  **Output Data (`example_data/output.csv`):**
    After running, the `output.csv` file will be created (or overwritten) with the original data plus the information retrieved and processed through the multi-step workflow:
    ```csv
    CompanyName,Region,ContactPerson,number_of_employees,annual_revenue,headquarters_location,year_founded,industry,website
    Apple Inc.,North America,,164000,391035000000.0,"Cupertino, California, United States",1976,Consumer Electronics,https://www.apple.com/
    Microsoft Corporation,North America,,228000,245122000000.0,"Redmond, Washington",1975,Information technology,microsoft.com
    Alphabet Inc.,North America,,183323,350018000000.0,"Mountain View, California, United States",2015,Conglomerate,https://abc.xyz
    "Amazon.com, Inc.",North America,,1556000,637959000000.0,"Seattle, Washington and Arlington, Virginia, U.S.",1994,"Conglomerate (E-commerce, Cloud Computing, Online Advertising, Digital Streaming, Artificial Intelligence, Retail)",https://www.amazon.com
    "Meta Platforms, Inc.",North America,,74067,164500000000.0,"Menlo Park, California",2004,Social media,meta.com
    "Cisco Systems, Inc.",North America,,90400,53803000000.0,"San Jose, California, U.S.",1984,"Networking hardware, Networking software, Communication Equipment, IT Services",cisco.com
    ```

## License

This project is licensed under the MIT License - see the `pyproject.toml` file for details (implicitly, as no separate LICENSE file exists).
