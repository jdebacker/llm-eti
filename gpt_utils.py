from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def get_gpt4_response(prompt, n=1):
    response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}], n=n
    )
    l = []
    for i in range(n):
        l.append(response.choices[i].message.content)
    return l
