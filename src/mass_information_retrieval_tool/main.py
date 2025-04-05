import argparse
import logging
import sys
from mass_information_retrieval_tool.core import (
    load_config,
    configure_gemini,
    process_csv
)

# Configure logging (can be configured here or rely on core.py's config)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Retrieves structured information for items in a CSV file using Google Gemini."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)"
    )
    args = parser.parse_args()

    try:
        # 1. Load Configuration
        config = load_config(args.config)

        # 2. Configure Gemini Client
        configure_gemini(config['api_key_env_var'])

        # 3. Process the CSV
        process_csv(config)

        logging.info("Processing complete.")

    except FileNotFoundError:
        # Error already logged in load_config
        sys.exit(1)
    except (ValueError, KeyError) as e:
        # Catch config validation errors or missing API key errors
        logging.error(f"Configuration or Setup Error: {e}")
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors during processing
        logging.error(f"An unexpected error occurred: {e}", exc_info=True) # Include traceback
        sys.exit(1)

if __name__ == "__main__":
    main()
