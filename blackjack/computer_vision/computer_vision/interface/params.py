import os
from dotenv import load_dotenv

load_dotenv()

TRAIN_DIR_IMGS = os.environ.get("TRAIN_DIR_IMGS")
TRAIN_DIR_LABELS = os.environ.get("TRAIN_DIR_LABELS")

VALID_DIR_IMGS = os.environ.get("VALID_DIR_IMGS")
VALID_DIR_LABELS = os.environ.get("VALID_DIR_LABELS")

TEST_DIR_IMGS = os.environ.get("TEST_DIR_IMGS")
TEST_DIR_LABELS = os.environ.get("TEST_DIR_LABELS")

MODEL_PATH = os.environ.get("MODEL_PATH")
