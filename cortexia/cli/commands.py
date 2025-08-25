"""CLI commands for Cortexia Video SDK"""

import argparse
import sys
from pathlib import Path

from ..api.cortexia import Cortexia
from ..api.exceptions import CortexiaError


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI commands"""
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
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process video with features")
    process_parser.add_argument("video_path", type=str, help="Path to video file")
    process_parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        help="Features to use (detection, segmentation, description, listing, feature_extraction)",
        default=["detection"],
    )
    process_parser.add_argument(
        "--output", type=str, help="Output directory", default="output"
    )
    
    # Individual feature commands
    features = ["detection", "segmentation", "description", "listing", "feature_extraction"]
    for feature in features:
        feature_parser = subparsers.add_parser(feature, help=f"Run {feature} feature")
        feature_parser.add_argument("video_path", type=str, help="Path to video file")
        feature_parser.add_argument(
            "--output", type=str, help="Output directory", default="output"
        )
    
    # List features command
    list_parser = subparsers.add_parser("list-features", help="List available features")
    
    return parser


def handle_process_command(args) -> int:
    """Handle process command"""
    try:
        cortexia = Cortexia.from_config(args.config)
        results = cortexia.process_video_with_features(args.video_path, args.features)
        print(f"Processing complete with features: {args.features}")
        return 0
    except CortexiaError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def handle_feature_command(args, feature_name: str) -> int:
    """Handle individual feature command"""
    try:
        cortexia = Cortexia.from_config(args.config)
        feature = cortexia.get_feature(feature_name)
        print(f"Running {feature_name} on {args.video_path}")
        # Implementation will depend on video loading utilities
        return 0
    except CortexiaError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def handle_list_features_command(args) -> int:
    """Handle list-features command"""
    try:
        cortexia = Cortexia.from_config(args.config)
        features = cortexia.list_features()
        print("Available features:")
        for feature in features:
            print(f"  - {feature}")
        return 0
    except CortexiaError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == "process":
        return handle_process_command(args)
    elif args.command == "list-features":
        return handle_list_features_command(args)
    elif args.command in ["detection", "segmentation", "description", "listing", "feature_extraction"]:
        return handle_feature_command(args, args.command)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())