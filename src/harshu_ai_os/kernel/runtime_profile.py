from dataclasses import dataclass

from harshu_ai_os.kernel.config import get_app_mode


@dataclass
class RuntimeProfile:
    system_name: str
    mode: str
    version: str = "0.1.0"

    def show_summary(self) -> str:
        return f"System: {self.system_name}\nMode: {self.mode}\nVersion: {self.version}"


if __name__ == "__main__":
    app_mode = get_app_mode()

    harshu = RuntimeProfile(
        system_name="Harshu AI OS",
        mode=app_mode,
    )

    print(harshu.show_summary())