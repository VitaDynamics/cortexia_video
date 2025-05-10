import logging
from cortexia_video.config_manager import ConfigManager
from cortexia_video.processing_manager import ProcessingManager
from cortexia_video.logger_setup import setup_logging
import pretty_errors

logger = logging.getLogger(__name__)

def main():
    try:
        config_manager = ConfigManager(config_dir="config/", config_name="config") 
        config_manager.load_config()
        setup_logging(config_manager)
        input_video = config_manager.get_param('processing.input_video_path', 'sample_data/input.mp4')
        processing_mode = config_manager.get_param('processing.default_mode', 'detect_segment_describe')
        
        logger.info(f"Starting video processing for: {input_video} with mode: {processing_mode}")
        
        manager = ProcessingManager(config_manager=config_manager) # Pass initialized config_manager
        output_file = manager.process_video(input_video, processing_mode)
        
        if output_file:
            logger.info(f"Processing complete. Annotations saved to: {output_file}")
        else:
            logger.error("Processing failed. No output file was generated.")

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)

if __name__ == "__main__":
    main()