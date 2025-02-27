from api.app.schemas.shared.constants import LANGUAGE
from fastapi import status


class BaseException(Exception):
    output: dict

    def __init__(
        self,
        message: str,
        status_code: int,
        error_code: int = status.HTTP_400_BAD_REQUEST,
        language: str = LANGUAGE,
        **kwargs
    ):
        self.error_code = error_code
        self.status_code = status_code
        self.message = message
        self.output = {
            "message": message,
            "error_code": self.error_code,
            "status_code": self.status_code,
            "language": language,
        } | kwargs
        super().__init__(self.output)


class Forbidden(BaseException):
    messages = {
        "cz": "Zakázáno",
        "en": "Forbidden",
        "fr": "Interdit",
    }

    def __init__(self, error_code: int = 1, language: str = LANGUAGE, message: str | None = None):
        self.status_code = status.HTTP_403_FORBIDDEN
        if message:
            self.message = message
        else:
            self.message = self.messages.get(language, self.messages.get(LANGUAGE, ""))
        super().__init__(self.message, error_code=error_code, status_code=self.status_code, language=language)


class NotFound(BaseException):
    messages = {
        "cz": "Nenalezeno",
        "en": "Not found",
        "fr": "Pas trouvé",
    }

    def __init__(
        self, id: int | None = None, error_code: int = 2, language: str = LANGUAGE, message: str | None = None
    ):
        if message:
            self.message = message
        else:
            self.message = self.messages.get(language, self.messages.get(LANGUAGE, ""))
        self.status_code = status.HTTP_404_NOT_FOUND
        super().__init__(self.message, error_code=error_code, status_code=self.status_code, language=language, id=id)


class InvalidData(BaseException):
    messages = {
        "cz": "Neplatná data",
        "en": "Invalid data",
        "fr": "Données invalides",
    }

    def __init__(self, error_code: int = 3, language: str = LANGUAGE, message: str | None = None):
        self.status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        if message:
            self.message = message
        else:
            self.message = self.messages.get(language, self.messages.get(LANGUAGE, ""))
        super().__init__(self.message, self.status_code, error_code, language=language)


class NotCompatible(BaseException):
    messages = {
        "cz": "Nekompatibilní",
        "en": "Not compatible",
        "fr": "Non compatible",
    }

    def __init__(self, error_code: int = 4, language: str = LANGUAGE, message: str | None = None):
        self.status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        if message:
            self.message = message
        else:
            self.message = self.messages.get(language, self.messages.get(LANGUAGE, ""))
        super().__init__(self.message, error_code=error_code, status_code=self.status_code, language=language)


class AlreadyExists(BaseException):
    messages = {
        "cz": "Již existuje",
        "en": "Already exists",
        "fr": "Déjà existant",
    }

    def __init__(self, error_code: int = 5, language: str = LANGUAGE, message: str | None = None):
        self.status_code = status.HTTP_409_CONFLICT
        if message:
            self.message = message
        else:
            self.message = self.messages.get(language, self.messages.get(LANGUAGE, ""))
        super().__init__(self.message, error_code=error_code, status_code=self.status_code, language=language)


class NothingToUpdate(BaseException):
    messages = {
        "cz": "Nic k aktualizaci",
        "en": "Nothing to update",
        "fr": "Rien à mettre à jour",
    }

    def __init__(self, error_code: int = 6, language: str = LANGUAGE, message: str | None = None):
        self.status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        if message:
            self.message = message
        else:
            self.message = self.messages.get(language, self.messages.get(LANGUAGE, ""))
        super().__init__(self.message, error_code=error_code, status_code=self.status_code, language=language)


class InvalidCondition(InvalidData):
    messages = {
        "cz": "Neplatná podmínka",
        "en": "Invalid condition",
        "fr": "Condition invalide",
    }

    def __init__(self, error_code: int = 7, language: str = LANGUAGE, message: str | None = None):
        if message:
            self.message = message
        else:
            self.message = self.messages.get(language, self.messages.get(LANGUAGE, ""))
        super().__init__(message=self.message, error_code=error_code, language=language)


class InvalidOrder(InvalidData):
    messages = {
        "cz": "Neplatné řazení",
        "en": "Invalid order",
        "fr": "Ordre invalide",
    }

    def __init__(self, error_code: int = 8, language: str = LANGUAGE, message: str | None = None):
        if message:
            self.message = message
        else:
            self.message = self.messages.get(language, self.messages.get(LANGUAGE, ""))
        super().__init__(message=self.message, error_code=error_code, language=language)


class InvalidName(InvalidData):
    messages = {
        "cz": "Neplatné jméno",
        "en": "Invalid name",
        "fr": "Nom invalide",
    }

    def __init__(self, error_code: int = 9, language: str = LANGUAGE, message: str | None = None):
        if message:
            self.message = message
        else:
            self.message = self.messages.get(language, self.messages.get(LANGUAGE, ""))
        super().__init__(message=self.message, error_code=error_code, language=language)


class NotEnabled(BaseException):
    messages = {
        "cz": "Není povoleno",
        "en": "Not enabled",
        "fr": "Non activé",
    }

    def __init__(self, error_code: int = 10, language: str = LANGUAGE, message: str | None = None):
        self.status_code = status.HTTP_423_LOCKED
        if message:
            self.message = message
        else:
            self.message = self.messages.get(language, self.messages.get(LANGUAGE, ""))
        super().__init__(message=self.message, error_code=error_code, status_code=self.status_code, language=language)


class InternalError(BaseException):
    messages = {
        "cz": "Interní chyba",
        "en": "Internal error",
        "fr": "Erreur interne",
    }

    def __init__(self, error_code: int = 11, language: str = LANGUAGE, message: str | None = None):
        self.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if message:
            self.message = message
        else:
            self.message = self.messages.get(language, self.messages.get(LANGUAGE, ""))
        super().__init__(message=self.message, error_code=error_code, status_code=self.status_code, language=language)


class InvalidID(BaseException):
    messages = {
        "cz": "Neplatné ID",
        "en": "Invalid ID",
        "fr": "ID invalide",
    }

    def __init__(self, error_code: int = 12, language: str = LANGUAGE, message: str | None = None):
        self.status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        if message:
            self.message = message
        else:
            self.message = self.messages.get(language, self.messages.get(LANGUAGE, ""))
        super().__init__(message=self.message, error_code=error_code, status_code=self.status_code, language=language)

