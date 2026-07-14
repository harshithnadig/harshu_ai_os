import dotenv
import os

dotenv.load_dotenv()

class MissingConfigError(Exception):
    pass

def get_app_mode() -> str:
    app_mode = os.getenv("HARSHU_AI_OS_MODE")
    if app_mode is None:
        raise MissingConfigError("HARSHU_AI_OS_MODE is missing")
    return app_mode

if __name__ == "__main__":
    print(get_app_mode())