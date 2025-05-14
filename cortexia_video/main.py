import argparse
import logging
import os
import sys

from cortexia_video.config_manager import ConfigManager
from cortexia_video.logger_setup import setup_logging
from cortexia_video.processing_manager import ProcessingManager

logger = logging.getLogger(__name__)


# TODO: expeted to use decord
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cortexia Video - Video annotation framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (TOML, YAML, or JSON)",
        default="./config.toml",
    )

    return parser.parse_args()


def main():
    """Main entry point for the CLI application."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Check if the specified config file exists
        config_file_path = args.config
        if config_file_path and not os.path.isabs(config_file_path):
            # Convert relative path to absolute
            config_file_path = os.path.abspath(config_file_path)

        # Create ConfigManager with the specified config file
        config_manager = ConfigManager(config_file_path=config_file_path)

        # Load the configuration
        try:
            config_manager.load_config()
        except FileNotFoundError as e:
            logger.error(f"Configuration error: {e}")
            print(f"Error: {e}")
            print(f"Using default path: {config_file_path}")
            # If config file doesn't exist at the specified path, try the default location
            if config_file_path != os.path.join("config", "config.toml"):
                print("Trying fallback location: config/config.toml")
                config_manager = ConfigManager(
                    config_dir="config/", config_name="config"
                )
                config_manager.load_config()
            else:
                sys.exit(1)

        # Setup logging
        setup_logging(config_manager)

        # Get processing parameters
        input_video = config_manager.get_param(
            "processing.input_video_path", "sample_data/input.mp4"
        )
        processing_mode = config_manager.get_param(
            "processing.default_mode", "detect_segment_describe"
        )

        logger.info(
            f"Starting video processing for: {input_video} with mode: {processing_mode}"
        )

        # Process the video
        manager = ProcessingManager(config_manager=config_manager)
        output_file = manager.process_video(input_video, processing_mode)

        if output_file:
            logger.info(f"Processing complete. Annotations saved to: {output_file}")
            print(f"Processing complete. Annotations saved to: {output_file}")
        else:
            logger.error("Processing failed. No output file was generated.")
            print("Processing failed. No output file was generated.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
