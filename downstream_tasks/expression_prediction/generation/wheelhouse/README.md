# Build the wheelhouse

Run `build_wheels.sh` on a machine with access to the two package directories and their dependencies. Pass the **generation** directory first and the **gena_expression API** directory second:

```bash
bash ../scripts/build_wheels.sh \
  /path/to/generation \
  /path/to/expression_prediction/api
```

Replace the example paths with the paths on your machine. Each directory must be an installable Python project, with a `pyproject.toml` or `setup.py`. Quote paths that contain spaces.

The script runs `python3 -m pip wheel` for both projects and their dependencies. Wheels are saved in the same `wheelhouse` directory as the script, regardless of the directory from which it is run. Existing files are not removed. The build typically needs internet access to download dependencies and may need compilers or other build tools for packages without suitable prebuilt wheels.

To choose a different Python interpreter or virtual environment, set `PYTHON` to its executable:

```bash
PYTHON=/path/to/venv/bin/python bash /path/to/wheelhouse/build_wheels.sh /path/to/generation /path/to/api
```

Build with the same Python version, operating system, and CPU architecture as the remote server, and account for any server-specific accelerator dependencies. Wheels built on a different system may not be compatible. Check that the wheelhouse contains the expected packages before copying it to the server.
