import sys
import yaml

from tests.test_streamers import create_object


if __name__ == "__main__":
    config_path = sys.argv[1]

    with open(config_path) as f:
            config = yaml.load(f)

    streamer = create_object(config)
