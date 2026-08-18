# Hermes Cloud Telegram Agent — Render Deployment Guide ($0 Target)

This guide walks through deploying the **Hermes Cloud Telegram Agent** to **Render** using their genuinely free Web Service tier (**$0, NO Credit Card Required**).

---

## 1. Render Free Tier Overview

- **Cost**: **$0.00 / month**
- **Credit Card Required?**: **NO**
- **Free Allowance**: 750 free execution hours per month (sufficient for 1 continuous web service).
- **Service Type**: Web Service (Docker).
- **Memory**: 512 MB RAM.

---

## 2. Step-by-Step Deployment Instructions

### Step 1: Create a Free Render Account
1. Go to [Render Signup](https://dashboard.render.com/register).
2. Sign up with GitHub or email (**No credit card required**).

### Step 2: Create New Web Service
1. Click **New +** -> **Web Service**.
2. Connect your Git repository (GitHub/GitLab).
3. Select Environment: **Docker**.
4. Set Dockerfile Path: `cloud/Dockerfile.render` (or default `Dockerfile`).
5. Choose Plan: **Free ($0/mo)**.

### Step 3: Configure Environment Variables
In Render **Environment** settings, add the following secrets:

| Variable | Value | Description |
| :--- | :--- | :--- |
| `TELEGRAM_BOT_TOKEN` | `123456789:ABC...` | Bot token from `@BotFather` |
| `TELEGRAM_ALLOWED_USERS` | `123456789` | Your numeric Telegram ID from `@userinfobot` |
| `GEMINI_API_KEY` | `AIzaSy...` | Free API key from [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `DEFAULT_MODEL` | `gemini-2.5-flash` | Default LLM model |
| `TERMINAL_BACKEND` | `vercel_sandbox` | Remote sandbox execution backend |

---

## 3. Keep-Alive Configuration (Preventing Sleep for 24/7 Uptime)

Render's free tier spins down web services after 15 minutes of HTTP inactivity.

To keep Hermes active 24/7:
1. Create a free account on [UptimeRobot](https://uptimerobot.com) or [Cron-job.org](https://cron-job.org).
2. Add an HTTP monitor pointing to your Render Web Service URL:
   `https://your-service-name.onrender.com/`
3. Set ping interval to **every 10 minutes**.

The embedded health server in `cloud/cloud_start.py` will respond `200 OK` on `/`, keeping your Render service active 24/7 for $0 without sleeping!
