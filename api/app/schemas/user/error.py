from api.app.schemas.error.base_error import NotFound
from api.app.schemas.shared.constants import LANGUAGE


class UserNotFound(NotFound):
    messages = {
        "cz": "Uživatel nebyl nalezen",
        "en": "User not found",
        "fr": "Utilisateur non trouvé",
    }

    def __init__(self, language: str = LANGUAGE, message: str | None = None):
        if message:
            self.message = message
        else:
            self.message = self.messages.get(language, self.messages.get(LANGUAGE, ""))
        super().__init__(message=self.message)
