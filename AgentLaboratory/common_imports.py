"""Minimal shared imports for the AgentLaboratory orchestration path.

This module intentionally keeps the baseline RSS low. The repository's hot path
for remote/g4f orchestration mostly needs stdlib helpers plus a couple of small
optional libraries. Heavy scientific / ML stacks should be imported locally in
functions that truly need them.
"""

import argparse
import base64
import collections
import csv
import datetime
import glob
import gzip
import hashlib
import itertools
import json
import logging
import math
import multiprocessing
import os
import pathlib
import pickle
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
import uuid
import warnings
import zipfile
from functools import lru_cache, partial
from multiprocessing import Pool

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

__all__ = [name for name in globals() if not name.startswith("_")]
