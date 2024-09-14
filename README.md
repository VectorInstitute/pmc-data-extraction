# OpenPMC-VL

----------------------------------------------------------------------------------------

[![code checks](https://github.com/VectorInstitute/aieng-template/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/aieng-template/actions/workflows/code_checks.yml)
[![integration tests](https://github.com/VectorInstitute/aieng-template/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/aieng-template/actions/workflows/integration_tests.yml)
[![docs](https://github.com/VectorInstitute/aieng-template/actions/workflows/docs_deploy.yml/badge.svg)](https://github.com/VectorInstitute/aieng-template/actions/workflows/docs_deploy.yml)
[![codecov](https://codecov.io/gh/VectorInstitute/aieng-template/branch/main/graph/badge.svg)](https://codecov.io/gh/VectorInstitute/aieng-template)
[![license](https://img.shields.io/github/license/VectorInstitute/aieng-template.svg)](https://github.com/VectorInstitute/aieng-template/blob/main/LICENSE.md)

A toolkit to download, augment, and benchmark OpenPMC-VL; a large dataset of image-text pairs extracted from open-access scientific articles on PubMedCentral.

## 🧑🏿‍💻 Developing

### Installing dependencies

We use
[poetry](https://python-poetry.org/docs/#installation)
for dependency management. Please make sure it is installed.

Then create and activate your virtual environment by running
```bash
python3 -m venv <env_name>
source <env_name>/bin/activate
```
Finally, install all dependencies in the virtual environment by
```bash
python3 -m poetry install
```

In order to install dependencies for testing (codestyle, unit tests, integration tests),
run:
```bash
python3 -m poetry install --with test
```
