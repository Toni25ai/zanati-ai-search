import os, time, re, json
import numpy as np
from numpy.linalg import norm
from openai import OpenAI
from supabase import create_client, Client
from fastapi import FastAPI

# Konfigurim identik, performancë FAST, pa prekur cilësi
