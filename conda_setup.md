# Anaconda Setup Guide for FluxPipeline

This guide provides step-by-step instructions for setting up the FluxPipeline project using Anaconda. Anaconda helps manage Python environments and dependencies efficiently.

## Table of Contents

- [Anaconda Setup Guide for FluxPipeline](#anaconda-setup-guide-for-fluxpipeline)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installing Anaconda](#installing-anaconda)
    - [Windows Installation](#windows-installation)
    - [macOS Installation](#macos-installation)
    - [Linux Installation](#linux-installation)
  - [Creating the Project Environment](#creating-the-project-environment)
    - [Step 1: Open Anaconda Prompt or Terminal](#step-1-open-anaconda-prompt-or-terminal)
    - [Step 2: Create a New Environment](#step-2-create-a-new-environment)
    - [Step 3: Activate the Environment](#step-3-activate-the-environment)
  - [Installing Dependencies](#installing-dependencies)
    - [Option 1: CPU-only Installation](#option-1-cpu-only-installation)
    - [Option 2: NVIDIA GPUs (CUDA support)](#option-2-nvidia-gpus-cuda-support)
    - [Option 3: AMD GPUs (ROCm support)](#option-3-amd-gpus-rocm-support)
    - [Option 4: Intel GPUs (OneAPI support)](#option-4-intel-gpus-oneapi-support)
  - [Troubleshooting Common Issues](#troubleshooting-common-issues)
    - [Issue 1: Conda Command Not Found](#issue-1-conda-command-not-found)
    - [Issue 2: Conflicts During Dependency Installation](#issue-2-conflicts-during-dependency-installation)
    - [Issue 3: Environment Activation Fails](#issue-3-environment-activation-fails)
  - [Examples of Environment Commands](#examples-of-environment-commands)
    - [Listing All Environments](#listing-all-environments)
    - [Deactivating an Environment](#deactivating-an-environment)
    - [Removing an Environment](#removing-an-environment)
  - [Additional Tips](#additional-tips)
    - [Updating Conda](#updating-conda)
    - [Creating an Environment from a YAML File](#creating-an-environment-from-a-yaml-file)
  - [Further Assistance](#further-assistance)

---

## Prerequisites

- **Internet Connection:** Required to download Anaconda and project dependencies.
- **Administrator Rights:** May be needed for installation on some systems.

## Installing Anaconda

Download and install Anaconda appropriate for your operating system.

### Windows Installation

1. **Download Installer:**

   - Go to the [Anaconda Distribution](https://www.anaconda.com/products/distribution) page.
   - Click on **Download** for Windows.

2. **Run the Installer:**

   - Double-click the downloaded `.exe` file.
   - Follow the prompts in the setup wizard.

3. **Add Anaconda to PATH (Optional but Recommended):**

   - During installation, check the option **Add Anaconda to my PATH environment variable**.

   *Explanation:* This allows you to use `conda` commands from any command prompt.

4. **Complete Installation:**

   - Click **Install** and wait for the process to complete.

### macOS Installation

1. **Download Installer:**

   - Go to the [Anaconda Distribution](https://www.anaconda.com/products/distribution) page.
   - Click on **Download** for macOS.

2. **Run the Installer:**

   - Open the downloaded `.pkg` file.
   - Follow the installation prompts.

3. **Add Anaconda to PATH:**

   - During installation, you may need to run `conda init` in your terminal.

4. **Complete Installation:**

   - Finish the setup and close the installer.

### Linux Installation

1. **Download Installer:**

   - Go to the [Anaconda Distribution](https://www.anaconda.com/products/distribution) page.
   - Download the Linux installer (e.g., `Anaconda3-2023.07-Linux-x86_64.sh`).

2. **Run the Installer:**

   ```bash
   bash Anaconda3-2023.07-Linux-x86_64.sh
   ```

3. **Follow the Prompts:**

   - Review the license agreement.
   - Choose the installation location (default is usually fine).

4. **Initialize Conda:**

   - When prompted, type `yes` to initialize Conda.

## Creating the Project Environment

### Step 1: Open Anaconda Prompt or Terminal

- **Windows:** Open **Anaconda Prompt** from the Start Menu.
- **macOS/Linux:** Open your terminal application.

### Step 2: Create a New Environment

Create a new environment named `flux` with Python 3.11:

```bash
conda create -n flux python=3.11 -y
```

*Explanation:* This command creates a new environment called `flux` with the specified Python version.

### Step 3: Activate the Environment

Activate the newly created environment:

```bash
conda activate flux
```

*Example Output:*

```bash
(flux) C:\Users\YourName>
```

## Installing Dependencies

With the environment activated, install the required dependencies.

### Option 1: CPU-only Installation

```bash
pip install -r requirements.txt
```

*Explanation:* Installs all packages listed in `requirements.txt` for CPU-only operation.

### Option 2: NVIDIA GPUs (CUDA support)

Ensure you have the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed.

```bash
pip install -r requirements_cuda.txt
```

*Explanation:* Installs GPU-accelerated packages suitable for NVIDIA GPUs.

### Option 3: AMD GPUs (ROCm support)

Ensure you have the [AMD ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) platform installed.

```bash
pip install -r requirements_rocm.txt
```

*Explanation:* Installs GPU-accelerated packages suitable for AMD GPUs.

### Option 4: Intel GPUs (OneAPI support)

Ensure you have the [Intel oneAPI Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html) installed.

```bash
pip install -r requirements_intel.txt
```

*Explanation:* Installs GPU-accelerated packages suitable for Intel GPUs.

## Troubleshooting Common Issues

### Issue 1: Conda Command Not Found

**Solution:**

- Ensure Anaconda is added to your PATH.
- Alternatively, use the Anaconda Prompt (Windows) or run `source ~/anaconda3/bin/activate` (Linux/macOS).

### Issue 2: Conflicts During Dependency Installation

**Solution:**

- Update Conda and Pip:

  ```bash
  conda update -n base -c defaults conda
  pip install --upgrade pip
  ```

- Try creating a new environment.
- Check for version conflicts in the `requirements` files.

### Issue 3: Environment Activation Fails

**Solution:**

- Ensure the environment was created successfully.
- Check the list of environments:

  ```bash
  conda env list
  ```

- If the environment doesn't exist, recreate it.

## Examples of Environment Commands

### Listing All Environments

```bash
conda env list
```

*Example Output:*

```bash
# conda environments:
#
base                  *  /home/yourname/anaconda3
flux                     /home/yourname/anaconda3/envs/flux
```

### Deactivating an Environment

```bash
conda deactivate
```

*Explanation:* Returns you to the base environment.

### Removing an Environment

```bash
conda remove -n flux --all
```

*Warning:* This deletes the `flux` environment completely.

## Additional Tips

### Updating Conda

Keep Conda up to date to avoid issues:

```bash
conda update -n base -c defaults conda
```

### Creating an Environment from a YAML File

If a `environment.yml` file is provided:

```bash
conda env create -f environment.yml
```

*Explanation:* Creates an environment with all specified dependencies.

## Further Assistance

- **Conda Documentation:** [Managing Conda Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- **Anaconda Support:** [Anaconda Knowledge Base](https://support.anaconda.com/)
- **Contact:** For project-specific issues, refer to the [Contact](README.md#contact) section in the README.
