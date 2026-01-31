import streamlit as st
import time
import os

log_file = r"A:\projects\project3\PythonProject\status.txt"

st.title("📊 Model Monitoring Dashboard")

log_display = st.empty()  # ✅ Placeholder for logs


def read_logs():
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            return f.readlines()
    except FileNotFoundError:
        return ["No logs yet..."]


# ✅ Main loop to continuously refresh logs
while True:
    logs = read_logs()

    # ✅ Update the same UI element dynamically
    log_display.text_area("Training Logs", "".join(logs), height=400)

    # ✅ Rerun the script to refresh instantly
    st.experimental_rerun()

    time.sleep(1)  # 🔄 Update every second
