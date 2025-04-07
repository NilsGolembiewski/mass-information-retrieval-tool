# Mass Information Retrieval Tool

A Python tool that retrieves structured information for a list of items from a CSV file using the Google Gemini API based on a YAML configuration file.

### Information Source (Gemini API with Google Search)

This tool utilizes the Google Gemini API to gather information. Crucially, it enables the **Google Search tool** within the API call. This allows the Gemini model to potentially perform live web searches via Google Search to supplement its internal knowledge base and provide more current or specific information when answering prompts. The quality and recency of the information depend on both the Gemini model's capabilities and the search results it accesses.

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

3.  **Set up Google Gemini API Key:**
    *   Create a file named `.env` in the project root directory (`/home/nils/Projects/mass-information-retrieval-tool/.env`).
    *   Add your Google Gemini API key to the `.env` file. The key name should match the `api_key_env_var` specified in your `config.yaml` (default is `GOOGLE_API_KEY`):
        ```dotenv
        GOOGLE_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY
        ```
    *   Ensure `.env` is listed in your `.gitignore` file to prevent committing secrets.

## Configuration (`config.yaml`)

The tool's behavior is controlled by a `config.yaml` file (default location is the project root). Create this file if it doesn't exist and configure the following options:

*   `input_csv`: (Required) Path to the input CSV file containing the items to look up.
*   `output_csv`: (Required) Path where the enriched output CSV file will be saved. The directory will be created if it doesn't exist.
*   `key_columns`: (Required) A list of column names in the `input_csv` that together identify the items (e.g., `["Company Name", "Product ID"]`) for which information should be retrieved.
*   `gemini_model`: (Required) The specific Google Gemini model to use (e.g., `gemini-1.5-flash`).
*   `api_key_env_var`: (Required) The name of the environment variable that holds your Google Gemini API key (defined in your `.env` file).
*   `schema`: (Required) A dictionary defining the desired output structure.
    *   Keys: The names of the new columns to be added to the output CSV.
    *   Values: The expected data type for each column (`str`, `int`, `float`, `bool`). The tool will attempt to convert the data received from Gemini to these types.
*   `prompt_template`: (Required) A template string for the prompt sent to Gemini.
    *   Use `{key_values_json}` as a placeholder for a JSON object containing the key-value pairs from the `key_columns` for the current row.
    *   Use `{schema_json}` as a placeholder for a JSON representation of the desired output structure (generated automatically based on the `schema`).

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
gemini_model: "gemini-1.5-flash"
api_key_env_var: "GOOGLE_API_KEY"

# Schema definition for the information to retrieve
schema:
  approx_employees: "int"
  industry: "str"
  headquarters_location: "str"
  founded_year: "int"

# Prompt template for querying Gemini
prompt_template: |
  For the entity described by the following key information:
  {key_values_json}

  Find the following details and provide them strictly as a JSON object matching this structure:
  {schema_json}

  If you cannot find a specific piece of information, use a JSON null value for that key. Do not add any explanatory text outside the JSON object.
```

## Usage

After installing dependencies with `poetry install`, you can run the tool using the script entry point:

```bash
poetry run mass-retrieve-info --config config.yaml
```

Alternatively, you can run it directly as a module:

```bash
poetry run python -m mass_information_retrieval_tool.main --config config.yaml
```

*   Replace `config.yaml` with the actual path to your configuration file if it's different or located elsewhere (the default is `config.yaml` in the current directory).
*   The script will read the input CSV, query Gemini for each item identified by the combination of values in the specified `key_columns`, attempt to parse and validate the results according to the `schema`, and save the enriched data to the `output_csv`.
*   Progress and any errors will be logged to the console.

## Example Usage (Using `example_data`)

This repository includes an `example_data` directory containing sample files to demonstrate the tool's functionality.

1.  **Configuration (`example_data/config.yaml`):**
    This file is pre-configured to use the example input/output files and defines a schema for company information:
    ```yaml
    # --- Configuration for Mass Information Retrieval Tool ---

    # Input CSV file path
    input_csv: "example_data/input.csv" # IMPORTANT: Update this path to your actual input file

# Output CSV file path (will be created)
output_csv: "example_data/output.csv" # IMPORTANT: Update this path if needed

# List of column names in the input CSV containing the key information
key_columns: ["CompanyName"] # IMPORTANT: Update this list with your actual column names

# Google Gemini API configuration
gemini_model: "gemini-2.5-pro-preview-03-25" # Using 2.5 Pro Preview model
    api_key_env_var: "GEMINI_API_KEY" # Name of the environment variable holding the API key

    # Parallel processing configuration
    max_parallel_requests: 5 # Maximum number of concurrent API requests (adjust as needed)

    # Schema definition for the information to retrieve
    # Keys are the desired output column names, values are the expected data types (str, int, float, bool)
    schema:
      number_of_employees: int
      annual_revenue: float
      headquarters_location: str
      year_founded: int
      industry: str
      website: str
      # Add more fields as needed

# Prompt template for querying Gemini
# Use {key_values_json} as a placeholder for the JSON object containing the key-value pairs from the key_columns
# Use {schema_json} as a placeholder for the desired JSON output structure
prompt_template: |
  For the entity described by the following key information:
  {key_values_json}

  Find the following details and provide them strictly as a JSON object matching this structure:
  {schema_json}

  If you cannot find a specific piece of information, use a JSON null value for that key. Do not add any explanatory text outside the JSON object.
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
    Execute the tool using the example configuration (using the script entry point):
    ```bash
    poetry run mass-retrieve-info --config example_data/config.yaml
    ```
    *(Ensure your `GEMINI_API_KEY` is set in the `.env` file as described in the Setup section).*

4.  **Output Data (`example_data/output.csv`):**
    After running, the `output.csv` file will be created (or overwritten) with the original data plus the information retrieved based on the schema:
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
