#!/usr/bin/env python3
try:
    import openai_utils
    print('openai_utils import successful')
    print('ChatGPT:', openai_utils.ChatGPT)
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()
