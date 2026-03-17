# Deploying AI Document Summarizer to Render

Render is an excellent platform for deploying this project. Since this application has 3 components (the Endee Database, the Python Backend, and the Python Frontend), the cleanest way to deploy it on Render is by using a `render.yaml` file, which deploys all three parts simultaneously as an Environment.

Here are the exact steps to deploy the project to Render for free.

---

### Step 1: Add a `render.yaml` Configuration File
I have created a `render.yaml` file in your repository. This file tells Render how to build and connect the database, the backend, and the frontend.

Make sure you commit and push the project to your GitHub repository:
```bash
git add .
git commit -m "Add Render deployment setup"
git push origin main
```

### Step 2: Create a Blueprint on Render
1. Go to your [Render Dashboard](https://dashboard.render.com).
2. Click the **New +** button and select **Blueprint**.
3. Connect your GitHub account and select your `shekhar-narayan-mishra/endee` repository.
4. Render will automatically detect the `render.yaml` file.
5. Click **Apply Blueprint**.

### Step 3: Add Your Groq API Key
While the Blueprint is provisioning, you must add your LLM API Key:
1. Go to the **Dashboard** and click on your **AI-Backend-API** Web Service.
2. Go to **Environment**.
3. Click **Add Environment Variable**.
4. Key: `GROQ_API_KEY` | Value: `your_actual_groq_api_key_here`.
5. Save changes.

---

## How it works on Render

The `render.yaml` creates three secure, isolated services that talk to each other automatically:

1. **Endee Database (Private Service)**: Runs the vector DB completely isolated from the internet on port `8080`.
2. **FastAPI Backend (Private Service)**: Runs your embedding and RAG logic natively. It connects to the internal Endee Database securely and accesses the internet to call Groq.
3. **Gradio Frontend (Web Service)**: A public-facing UI that securely routes the user's buttons and uploads right to the internal backend.

*Please note: Render's free tier spins down instances after 15 minutes of inactivity. The first request after a pause might take 30-60 seconds while the servers wake up.*
