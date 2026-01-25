import importlib.metadata
for pkg in ["langchain", "langchain-core", "langchain-huggingface", "langchain-community"]:
    try:
        print(f"{pkg}: {importlib.metadata.version(pkg)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg}: Not installed")
