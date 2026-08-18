# Iris Bot 👁️✨

> **Iris Bot** is a self-improving, multi-modal AI agent and Telegram gateway built on top of the Hermes Agent architecture. Designed for 24/7 cloud operation ($0 free-tier ready), serverless sandbox execution, web search, image generation, and Composio tool integrations.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![Platform: Telegram](https://img.shields.io/badge/Platform-Telegram-0088cc.svg)](https://core.telegram.org/bots/api)
[![Cloud Ready](https://img.shields.io/badge/Deployment-Render%20%7C%20HuggingFace%20%7C%20Vercel-purple.svg)](#cloud-deployment)

---

## 🌟 Key Features

- 🤖 **24/7 Telegram Gateway**: Interactive AI assistant responding to direct messages and group chats with full conversation history, voice message transcription, and media generation.
- ☁️ **$0 Cloud Ready**: Built-in HTTP health check server and automated keep-alive mechanisms for zero-cost 24/7 deployment on Render free web services or Hugging Face Spaces.
- 📊 **Web Dashboard**: Modern, glassmorphic monitoring dashboard (`apps/dashboard/`) for cloud status, session analytics, and task tracking.
- 🔌 **Tool & Integration Suite**:
  - **Composio SDK & V3 Actions**: Seamless access to hundreds of external apps and APIs.
  - **Multi-Model LLM Support**: Powered by Google Gemini (`gemini-2.5-flash`), OpenRouter, or custom endpoints.
  - **Image & Voice**: Dynamic image generation and speech-to-text processing.
- 🧠 **Persistent Memory & Learning**: Self-improving learning loop, automated skill creation, cross-session memory recall, and task routing.
- 🛡️ **Remote Sandboxes**: Execute terminal commands safely using Vercel Sandbox or Modal backends.

---

## 🏗️ Project Architecture

```
iris-bot/
├── api/                  # Serverless API handlers (cron jobs, webhooks)
├── apps/
│   └── dashboard/        # Web dashboard frontend (HTML/CSS/JS)
├── cloud/                # Cloud deployment manifests & bootstrapper
│   ├── cloud_start.py    # 24/7 Telegram bot & HTTP health server
│   ├── Dockerfile.hf     # Hugging Face Spaces Dockerfile
│   ├── Dockerfile.render # Render Web Service Dockerfile
│   ├── README_HF.md      # Hugging Face deployment guide
│   └── README_RENDER.md  # Render $0 deployment guide
├── lib/                  # Core modules (auth, Composio, Telegram, runner, memory)
├── scripts/              # Integration tests & cloud deployment scripts
├── supabase/             # Database schemas & migrations
├── run_agent.py          # Core AIAgent conversation loop
├── model_tools.py        # Tool discovery & function execution handler
└── requirements.txt      # Project dependencies
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- Git
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- Free API Key from [Google AI Studio](https://aistudio.google.com/)

### 2. Local Setup

```bash
# Clone the repository
git clone https://github.com/ashishnotfound/iris-bot.git
cd iris-bot

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.cloud.example` to `.env` and fill in your credentials:

```bash
cp .env.cloud.example .env
```

Set the essential environment variables:

```env
TELEGRAM_BOT_TOKEN="your-telegram-bot-token"
TELEGRAM_ALLOWED_USERS="your-numeric-telegram-chat-id"
GEMINI_API_KEY="your-gemini-api-key"
DEFAULT_MODEL="gemini-2.5-flash"
TERMINAL_BACKEND="vercel_sandbox"
```

### 4. Running the Agent

Start the cloud agent & Telegram gateway locally:

```bash
python cloud/cloud_start.py
```

---

## ☁️ Cloud Deployment ($0 Target)

### Deploying to Render Free Tier ($0/mo)

1. Create a free account on [Render](https://dashboard.render.com/).
2. Create a new **Web Service** and connect your GitHub repository `iris-bot`.
3. Select **Docker** environment and set Dockerfile path to `cloud/Dockerfile.render`.
4. Choose the **Free ($0/mo)** plan.
5. Add environment variables (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_USERS`, `GEMINI_API_KEY`).
6. Set up a 10-minute HTTP ping on [UptimeRobot](https://uptimerobot.com) to your Render service URL to prevent idle sleeping.

For full step-by-step instructions, see [cloud/README_RENDER.md](cloud/README_RENDER.md).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).