from fastapi import Request, Response


class OptionsMiddleware:
    async def __call__(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return Response(content="OK", status_code=200)
        response = await call_next(request)
        return response
