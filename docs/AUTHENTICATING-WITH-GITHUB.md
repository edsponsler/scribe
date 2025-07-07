# Authenticating with GitHub using SSH in VS Code (WSL Ubuntu)

This guide provides step-by-step instructions for setting up SSH key authentication for GitHub. Using an SSH key is more secure and convenient than using a password because you won't have to enter your credentials every time you interact with your remote repository.

These instructions are designed for a new user working within a VS Code terminal connected to a WSL (Windows Subsystem for Linux) Ubuntu environment.

---

### Step 1: Check for Existing SSH Keys

First, let's see if you already have an SSH key on your system.

1.  Open your WSL Ubuntu terminal.
2.  Run the following command to list files in your `.ssh` directory:
    ```bash
    ls -al ~/.ssh
    ```
3.  Look for a pair of files named something like `id_ed25519` and `id_ed25519.pub`. The `.pub` file is your public key.

*   If you see these files, you already have a key and can **skip to Step 3**.
*   If you get an error that the directory doesn't exist or you don't see these files, proceed to Step 2.

---

### Step 2: Generate a New SSH Key

If you don't have a key, you'll need to generate one. We'll use the `ed25519` algorithm, which is modern and secure.

1.  Run the `ssh-keygen` command in your terminal. **Replace `your_email@example.com` with the email address associated with your GitHub account.**
    ```bash
    ssh-keygen -t ed25519 -C "your_email@example.com"
    ```

2.  You will be prompted for a file location and a passphrase:
    *   **"Enter a file in which to save the key..."**: Press **Enter** to accept the default location.
    *   **"Enter passphrase (empty for no passphrase):"**: This is highly recommended. Type a secure passphrase and press **Enter**. This passphrase protects your private key. You will be asked to enter it again to confirm.

---

### Step 3: Add Your SSH Key to the ssh-agent

The `ssh-agent` is a background program that handles your SSH keys so you don't have to type your passphrase every time.

1.  Start the agent in the background:
    ```bash
    eval "$(ssh-agent -s)"
    ```

2.  Add your new private key to the agent:
    ```bash
    ssh-add ~/.ssh/id_ed25519
    ```
    You will be prompted to enter the passphrase you created in Step 2.

---

### Step 4: Add the Public SSH Key to Your GitHub Account

Now, you need to give GitHub your **public key** so it can recognize your machine.

1.  Display your public key in the terminal and copy it to your clipboard.
    ```bash
    cat ~/.ssh/id_ed25519.pub
    ```
    This will print your key, which starts with `ssh-ed25519...`. **Copy the entire line of text.**

2.  Open your web browser and navigate to GitHub.com.
    *   Click on your profile picture in the top-right corner, then go to **Settings**.
    *   In the left sidebar, click on **SSH and GPG keys**.
    *   Click the **New SSH key** button.

3.  Add your key:
    *   **Title:** Give your key a descriptive name (e.g., "VS Code WSL Ubuntu").
    *   **Key type:** Keep it as "Authentication Key".
    *   **Key:** Paste the public key you copied from your terminal into this box.
    *   Click **Add SSH key**. You may be asked to re-enter your GitHub password to confirm.

---

### Step 5: Test Your SSH Connection

Finally, let's verify that everything is working correctly.

1.  In your WSL terminal, run this command:
    ```bash
    ssh -T git@github.com
    ```
2.  You may see a message about the authenticity of the host. This is normal. Type **`yes`** and press **Enter**.
3.  If successful, you will see a welcome message: `Hi <your-username>! You've successfully authenticated, but GitHub does not provide shell access.`

You are now fully configured to use Git commands like `git clone`, `git push`, and `git pull` securely from your terminal.