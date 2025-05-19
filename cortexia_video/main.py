import argparse
import logging
import os
import sys

from cortexia_video.config_manager import ConfigManager
from cortexia_video.logger_setup import setup_logging
from cortexia_video.processing_manager import ProcessingManager

logger = logging.getLogger(__name__)


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
        default="config/example_config.toml",
    )


    parser.add_argument(
        "--batch_mode",
        type=bool,
        help="Batch mode for processing videos",
        default=False,
    )

    return parser.parse_args()


def iterate_video(videos_dir: str):
    """Iterate over all videos in the given directory."""
    for video_file in os.listdir(videos_dir):
        if video_file.endswith(".mp4") or video_file.endswith(".avi") or video_file.endswith(".mov") or video_file.endswith(".mkv"):
            yield os.path.join(videos_dir, video_file)


def main():
    """Main entry point for the CLI application."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Create ConfigManager with the specified config file
        # parse as absolute path
        config_file_path = os.path.abspath(args.config)
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
        manager.load_components(processing_mode=processing_mode)
        for video_file in iterate_video(videos_dir=manager.config_manager.get_param("processing.input_video_path")):
            output_file = manager.process_video(video_path=video_file, processing_mode=processing_mode)

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

