## How to compile a project

```bash
make
```

To comile on

- `Linux` (Ubuntu families) you need
    - `python3` - for compile on Python3

    ```bash
    sudo apt update
    sudo apt install python3
    ```

    - `make` - for Makefile

    ```bash
    sudo apt update
    sudo apt install make
    ```

- `Windows`
    - `Makefile`

    Install make:

    ```powershell
    choco install make
    ```

    If you don't have choco, install it:

    ```powershell
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
    ```

    - `Python`

    Download from the official site