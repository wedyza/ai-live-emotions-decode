from contextlib import contextmanager

from fastapi import FastAPI

@contextmanager
def lifecycle():
    app = FastAPI()
    print("Start of work")
    yield
    print("End of work")

if __name__ == 'main':