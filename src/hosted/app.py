"""Independent hosted ASGI factory: uvicorn src.hosted.app:create_hosted_app --factory."""
from pathlib import Path
import importlib.util
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from .auth import Principal, bearer_token, require_principal
from .schemas import DatasetUpload, JobRequest, ProjectRequest, ProjectUpdate, SessionRequest
from .services import save_dataset, validate_job_config
from .settings import HostedSettings
from .store import HostedStore
from version import __version__


def client_ip(request, trusted_proxies=()):
    """Peer address; X-Forwarded-For is honoured only from configured proxies."""
    peer = request.client.host if request.client else 'unknown'
    if peer not in trusted_proxies:
        return peer
    # Walk right to left past our own proxies; the first other hop is the client.
    hops = [hop.strip() for hop in request.headers.get('x-forwarded-for', '').split(',') if hop.strip()]
    for hop in reversed(hops):
        if hop not in trusted_proxies:
            return hop
    return peer


def create_hosted_app(settings=None):
    settings = settings or HostedSettings.from_env()
    app = FastAPI(title='Merlina Wishcraft Hosted', version=__version__)
    app.state.settings = settings
    app.state.store = HostedStore(settings.database_path)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_credentials=False, allow_methods=['GET', 'POST', 'PATCH', 'DELETE'],
                           allow_headers=['Authorization', 'Content-Type'])

    @app.middleware('http')
    async def secure_responses(request, call_next):
        # Reject oversized JSON before buffering, including clients using chunked encoding.
        if request.method in ('POST', 'PUT', 'PATCH'):
            limit = settings.max_dataset_bytes * 2 + 65536
            try:
                declared = int(request.headers.get('content-length', '0'))
            except ValueError:
                return Response(status_code=400)
            if declared > limit:
                return Response(status_code=413)
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > limit:
                    return Response(status_code=413)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    @app.get('/health')
    def health():
        return {'status': 'ok', 'mode': 'hosted'}

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        # Provider/worker exception strings may contain credentials or user data.
        return JSONResponse(status_code=500, content={'detail': 'An internal error occurred'})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Do not echo rejected credentials, dataset contents or arbitrary inputs.
        return JSONResponse(status_code=422, content={'detail': [
            {'loc': error['loc'], 'msg': error['msg'], 'type': error['type']}
            for error in exc.errors()]})

    prefix = '/api/hosted'

    @app.post(prefix + '/session')
    def login(body: SessionRequest, request: Request):
        return app.state.store.login(body.email, body.password, settings.session_ttl_seconds,
                                     client_ip(request, settings.trusted_proxies))

    @app.get(prefix + '/session')
    def session(principal: Principal = Depends(require_principal)):
        return {'id': principal.account_id, 'email': principal.email}

    @app.delete(prefix + '/session', status_code=204)
    def logout(request: Request, principal: Principal = Depends(require_principal)):
        app.state.store.logout(bearer_token(request))
        return Response(status_code=204)

    @app.get(prefix + '/projects')
    def projects(principal: Principal = Depends(require_principal)):
        return app.state.store.list_projects(principal.account_id)

    @app.post(prefix + '/projects', status_code=201)
    def create_project(body: ProjectRequest, principal: Principal = Depends(require_principal)):
        return app.state.store.create_project(principal.account_id, body.name, body.goal)

    @app.get(prefix + '/projects/{project_id}')
    def project(project_id: str, principal: Principal = Depends(require_principal)):
        return app.state.store.require_project(principal.account_id, project_id)

    @app.patch(prefix + '/projects/{project_id}')
    def update_project(project_id: str, body: ProjectUpdate, principal: Principal = Depends(require_principal)):
        return app.state.store.update_project(principal.account_id, project_id, **body.model_dump(exclude_unset=True))

    @app.get(prefix + '/projects/{project_id}/datasets')
    def datasets(project_id: str, principal: Principal = Depends(require_principal)):
        return app.state.store.list_datasets(principal.account_id, project_id)

    @app.post(prefix + '/projects/{project_id}/datasets', status_code=201)
    def upload(project_id: str, body: DatasetUpload, principal: Principal = Depends(require_principal)):
        return save_dataset(app.state.store, settings, principal.account_id, project_id, body.name, body.content)

    @app.get(prefix + '/projects/{project_id}/jobs')
    def jobs(project_id: str, principal: Principal = Depends(require_principal)):
        return app.state.store.list_jobs(principal.account_id, project_id)

    @app.post(prefix + '/projects/{project_id}/jobs', status_code=201)
    def create_job(project_id: str, body: JobRequest, principal: Principal = Depends(require_principal)):
        clean = validate_job_config(app.state.store, principal.account_id, project_id, body.config.model_dump(exclude_none=True))
        return app.state.store.create_job(principal.account_id, project_id, clean)

    @app.get(prefix + '/projects/{project_id}/jobs/{job_id}')
    def job(project_id: str, job_id: str, principal: Principal = Depends(require_principal)):
        return app.state.store.require_job(principal.account_id, project_id, job_id)

    @app.get(prefix + '/projects/{project_id}/events')
    def events(project_id: str, principal: Principal = Depends(require_principal)):
        return app.state.store.list_events(principal.account_id, project_id)

    @app.get(prefix + '/projects/{project_id}/jobs/{job_id}/events')
    def job_events(project_id: str, job_id: str, principal: Principal = Depends(require_principal)):
        return app.state.store.list_events(principal.account_id, project_id, job_id)

    @app.get(prefix + '/projects/{project_id}/jobs/{job_id}/artifacts')
    def artifacts(project_id: str, job_id: str, principal: Principal = Depends(require_principal)):
        return app.state.store.list_artifacts(principal.account_id, project_id, job_id)

    @app.get(prefix + '/projects/{project_id}/jobs/{job_id}/artifacts/{artifact_id}/download')
    def download(project_id: str, job_id: str, artifact_id: str, principal: Principal = Depends(require_principal)):
        artifact = app.state.store.require_artifact(principal.account_id, project_id, job_id, artifact_id)
        downloader = getattr(app.state, 'artifact_download', None)
        if downloader is None:
            raise HTTPException(503, 'Artifact downloads are not configured')
        return downloader(artifact)

    # Each optional stack layer contributes only authenticated hosted routes.
    for module_name in ('routes_billing', 'routes_advisor'):
        if importlib.util.find_spec(f'src.hosted.{module_name}'):
            module = __import__(f'src.hosted.{module_name}', fromlist=['router'])
            app.include_router(module.router)

    frontend = Path(__file__).resolve().parents[2] / 'frontend' / 'hosted'
    if frontend.is_dir():
        app.mount('/hosted-static', StaticFiles(directory=frontend), name='hosted-static')

        @app.get('/', include_in_schema=False)
        def index():
            return FileResponse(frontend / 'index.html')

    return app
