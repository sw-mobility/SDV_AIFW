# Folders Explanation (children of api folder)

## Importation hierarchy by folders (Top to bottom)
    core, models, utils, codebase (Used to form service functions)
    |
    services (Used to form routes)
    |
    routes (Defines actual routes)
    |
    main (Aggregates routes to serve)

- .py files in the folders are primarily sorted by its functional difference, such as:
    - dataset
    - project
    - etc.

- They are then sorted again by its kind, such as:
    - raw
    - labeled
    - common
    - etc.

## Detailed Explanation by files

### codebase
    - Contains default codebases to be loaded when system starts.
    - Current codebase files are dummies.
    - Codebases will be 3 or more depending on function. Needs to be defined.
    - Those files can't be changed.

### core
- *config.py*
    - Contains..
        1. Core parameter sets to initialize DB containers
        2. Other settings such as MIME types, API_WORKDIR, etc. to avoid hardcoding fatal params.

- *minio.py*
    - Contains..
        1. MinIO Client which is capable of initialization and all basic I/O tasks
        - *The client is usually called in service functions*

- *mongodb.py*
    - Contains..
        1. MongoDB Client which is capable of initialization and all basic I/O tasks
        - *The client is usually called in service functions*

## models
### *dataset*
- *common_model.py*
    - Contains..
- *labeled_model.py*
    - Contains..
- *raw_model.py*
    - Contains..



safeafefawdawdadadazczczxvdv


  