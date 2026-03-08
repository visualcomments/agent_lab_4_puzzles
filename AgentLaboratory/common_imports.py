# General-purpose imports
import os
import sys
import json
import time
import re
import math
import logging
import random
import shutil
import pathlib
import argparse
import itertools
import datetime
import collections
import subprocess
import importlib

# Data manipulation and analysis
import csv
import sqlite3
import pickle
import gzip
import zipfile
import tarfile
import glob
import hashlib
import uuid
import base64
import warnings
import multiprocessing
from multiprocessing import Pool
from functools import partial, lru_cache


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _safe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


HEAVY_IMPORTS_ENABLED = _truthy_env("AGENTLAB_HEAVY_IMPORTS", default=False)

pd = _safe_import("pandas")
np = _safe_import("numpy")
yaml = _safe_import("yaml")
h5py = _safe_import("h5py")

plt = None
sns = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import seaborn as sns
except Exception:
    sns = None

if HEAVY_IMPORTS_ENABLED:
    try:
        import plotly.express as px
    except Exception:
        px = None
    try:
        import plotly.graph_objects as go
    except Exception:
        go = None
else:
    px = None
    go = None

transformers = _safe_import("transformers")
torch = _safe_import("torch")
if torch is not None:
    try:
        import torch.nn as nn
        import torch.optim as optim
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset, random_split
    except Exception:
        nn = optim = F = DataLoader = Dataset = random_split = None
else:
    nn = optim = F = DataLoader = Dataset = random_split = None

if HEAVY_IMPORTS_ENABLED:
    tf = _safe_import("tensorflow")
else:
    tf = None

tiktoken = _safe_import("tiktoken")
nltk = _safe_import("nltk")
if nltk is not None:
    try:
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer, WordNetLemmatizer
    except Exception:
        word_tokenize = sent_tokenize = stopwords = PorterStemmer = WordNetLemmatizer = None
else:
    word_tokenize = sent_tokenize = stopwords = PorterStemmer = WordNetLemmatizer = None

if HEAVY_IMPORTS_ENABLED:
    spacy = _safe_import("spacy")
    sacremoses = _safe_import("sacremoses")
else:
    spacy = None
    sacremoses = None

if HEAVY_IMPORTS_ENABLED:
    diffusers = _safe_import("diffusers")
    if diffusers is not None:
        try:
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        except Exception:
            StableDiffusionPipeline = DPMSolverMultistepScheduler = None
    else:
        StableDiffusionPipeline = DPMSolverMultistepScheduler = None
else:
    diffusers = None
    StableDiffusionPipeline = DPMSolverMultistepScheduler = None

if HEAVY_IMPORTS_ENABLED:
    accelerate = _safe_import("accelerate")
    if accelerate is not None:
        try:
            from accelerate import Accelerator
        except Exception:
            Accelerator = None
    else:
        Accelerator = None
else:
    accelerate = None
    Accelerator = None

if HEAVY_IMPORTS_ENABLED:
    huggingface_hub = _safe_import("huggingface_hub")
    if huggingface_hub is not None:
        try:
            from huggingface_hub import HfApi, notebook_login
        except Exception:
            HfApi = notebook_login = None
    else:
        HfApi = notebook_login = None
else:
    huggingface_hub = None
    HfApi = notebook_login = None

sklearn = _safe_import("sklearn")
if sklearn is not None:
    try:
        from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.svm import SVC
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
    except Exception:
        train_test_split = GridSearchCV = RandomizedSearchCV = None
        accuracy_score = precision_score = recall_score = f1_score = classification_report = confusion_matrix = None
        StandardScaler = MinMaxScaler = LabelEncoder = PCA = KMeans = SVC = None
        TfidfVectorizer = CountVectorizer = linear_kernel = cosine_similarity = None
else:
    train_test_split = GridSearchCV = RandomizedSearchCV = None
    accuracy_score = precision_score = recall_score = f1_score = classification_report = confusion_matrix = None
    StandardScaler = MinMaxScaler = LabelEncoder = PCA = KMeans = SVC = None
    TfidfVectorizer = CountVectorizer = linear_kernel = cosine_similarity = None

scipy = _safe_import("scipy")
if scipy is not None:
    try:
        from scipy import stats, signal, spatial
        from scipy.optimize import minimize
        from scipy.spatial.distance import euclidean, cosine
        from scipy.linalg import svd, eig
    except Exception:
        stats = signal = spatial = minimize = euclidean = cosine = svd = eig = None
else:
    stats = signal = spatial = minimize = euclidean = cosine = svd = eig = None

if HEAVY_IMPORTS_ENABLED:
    statsmodels_api = _safe_import("statsmodels.api")
    if statsmodels_api is not None:
        try:
            from statsmodels.api import OLS, Logit
            from statsmodels.tsa.arima_model import ARIMA
            from statsmodels.tsa.stattools import adfuller, pacf, acf
        except Exception:
            OLS = Logit = ARIMA = adfuller = pacf = acf = None
    else:
        OLS = Logit = ARIMA = adfuller = pacf = acf = None
else:
    OLS = Logit = ARIMA = adfuller = pacf = acf = None

try:
    from PIL import Image
except Exception:
    Image = None
imageio = _safe_import("imageio")
if HEAVY_IMPORTS_ENABLED:
    try:
        from skimage import io, color, filters, transform, exposure
    except Exception:
        io = color = filters = transform = exposure = None
else:
    io = color = filters = transform = exposure = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

pydantic = _safe_import("pydantic")
requests = _safe_import("requests")
aiohttp = _safe_import("aiohttp")

__all__ = [name for name in globals() if not name.startswith("_")]
