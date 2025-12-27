# Deployment Guide for Render.com

Since your project uses Django and Deep Learning models (OpenCV, NumPy), standard static hosting like Netlify will not work. We will use **Render**, which is designed for Python applications.

I have already configured your project with the necessary files (`render.yaml`, `build.sh`, `Procfile`, `requirements.txt`).

---

## Phase 1: Push Your Code to GitHub

1.  **Log in to GitHub** and create a **New Repository**.
    *   Name it `emotion-recognition-app`.
    *   Make it **Public** (or Private, up to you).
    *   **Do not** add a README or .gitignore (you already have files).
    *   Click **Create repository**.

2.  **Upload your code** using the terminal/command prompt.
    Open your terminal in the project folder (`d:\Project\...\Code\EGGHumanEmotion`) and run these commands one by one:

    ```bash
    # Initialize git if you haven't already (skip if already a git repo)
    git init

    # Add all files to the staging area
    git add .

    # Commit the changes
    git commit -m "Ready for deployment"

    # Link to your new GitHub repository (replace URL with YOUR repo URL)
    git remote add origin https://github.com/YOUR_USERNAME/emotion-recognition-app.git

    # Push the code
    git push -u origin main
    ```
    *(If `git push` fails because of a branch name, try `git push -u origin master`)*

---

## Phase 2: Deploy on Render

1.  **Create a Render Account**: Go to [dashboard.render.com](https://dashboard.render.com/) and sign up (you can use your GitHub account).

2.  **Create New Web Service**:
    *   Click the **"New +"** button in the top right.
    *   Select **"Web Service"**.

3.  **Connect GitHub**:
    *   You will see a list of your GitHub repositories.
    *   Find `emotion-recognition-app` and click **"Connect"**.

4.  **Configure & Deploy**:
    *   Render will detect the `render.yaml` file I created.
    *   You shouldn't need to change anything.
    *   **Build Command**: `./build.sh`
    *   **Start Command**: `gunicorn EGGHumanEmotion.wsgi:application`
    *   Select the **Free** instance type.
    *   Click **"Create Web Service"** or **"Deploy"**.

5.  **Wait for Build**:
    *   Render will now install your dependencies (this may take 5-10 minutes because of OpenCV).
    *   Watch the logs. Once it says "Your service is live", click the URL provided (e.g., `https://emotion-recognition.onrender.com`).

---

## Troubleshooting

-   **Database**: This deployment uses SQLite. Your database will reset every time the app restarts. For a permanent database, you need to add a PostgreSQL database on Render.
-   **Build Failures**: If the build fails due to memory, it might be because `opencv-contrib-python` is too heavy. We might need to switch to `opencv-python-headless` in `requirements.txt`.
