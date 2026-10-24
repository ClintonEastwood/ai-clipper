import os
import sys

# The scripts import each other as top-level modules (e.g. `from
# talknet_speaker_detection import ...`), so tests import them the same way.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
