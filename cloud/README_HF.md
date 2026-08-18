---
title: Hermes Cloud Telegram Agent
emoji: ☤
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Hermes Cloud Telegram Agent

This is a continuous, 24/7 cloud-hosted instance of **Hermes Agent by Nous Research**, controlled primarily via **Telegram**.

## Setup Instructions

### 1. Configure Secrets in Hugging Face Space Settings

In your Hugging Face Space **Settings** -> **Variables and secrets**, add the following **Secrets**:

- `TELEGRAM_BOT_TOKEN`: Token obtained from Telegram `@BotFather`
- `TELEGRAM_ALLOWED_USERS`: Your numeric Telegram Chat ID (obtain via `@userinfobot`), e.g., `123456789`
- `GEMINI_API_KEY`: API Key from [Google AI Studio](https://aistudio.google.com/app/apikey) (Free Tier)
- `OPENROUTER_API_KEY`: (Optional) API Key from [OpenRouter](https://openrouter.ai/keys)
- `DEFAULT_MODEL`: (Optional) Default model, e.g. `gemini-2.5-flash` or `meta-llama/llama-3.3-70b-instruct:free`
- `TERMINAL_BACKEND`: (Optional) `vercel_sandbox` or `docker`

### 2. Control via Telegram

Send `/start` or `/status` to your bot in Telegram to initiate interaction!
