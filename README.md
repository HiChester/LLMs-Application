# LLMs-Application

Project Overview
1. MBTI-Personality-Test：
The project aims to simplify the traditional 16-type MBTI personality test, reducing it to a series of key questions designed to quickly identify a user's personality type. This test leverages advanced LLM models and AI tools to analyze user responses and provide a four-letter MBTI code representing their personality type.

Features:
LLM-based analysis: Uses an LLM to analyze responses and determine personality types efficiently.
Fast and user-friendly: Allows users to receive their MBTI type within minutes by answering a few critical questions.
Powered by NVIDIA API: Integrates with NVIDIA's LLM API for model interaction and personality analysis.
Structure:
MBTI-Personality-Test/: This folder contains all code related to the MBTI test, including the Python script and model configuration files.
README.md: A detailed explanation of how the MBTI test works and how to run it in the provided Colab environment.
main.py: The main Python script that handles the interaction with the NVIDIA API and processes user input.
utils.py: Helper functions for parsing user input and managing responses.


Installation and Usage:
(1) Clone the repository(bash):
git clone https://github.com/HiChester/LLMs-Application.git
cd LLMs-Application/MBTI-Personality-Test

(2) Install the necessary dependencies(bash):
!pip install langchain langchain_nvidia_ai_endpoints rich

(3) Set up your NVIDIA API key(bash):
import os
os.environ['NVIDIA_API_KEY'] = 'your_api_key_here'
