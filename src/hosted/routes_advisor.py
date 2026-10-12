"""Authenticated advisor routes. None can reserve credits or launch compute."""
from fastapi import APIRouter, Depends, Request
from pydantic import Field

from .auth import Principal, require_principal
from .schemas import StrictModel
from .advisor import AdvisorProvider, AdvisorStore, inspect_dataset

router = APIRouter(prefix='/api/hosted/projects/{project_id}', tags=['advisor'])


class MessageRequest(StrictModel):
    message: str = Field(min_length=1, max_length=8000)
    request_key: str = Field(min_length=8, max_length=100, pattern=r'^[a-zA-Z0-9_-]+$')
    job_id: str | None = Field(default=None, max_length=64)
    dataset_id: str | None = Field(default=None, max_length=64)


class EvaluationRequest(StrictModel):
    job_id: str = Field(min_length=1, max_length=64)
    criteria: str = Field(min_length=1, max_length=4000)
    observations: str = Field(min_length=1, max_length=8000)


def service(request):
    return AdvisorStore(request.app.state.store)


@router.get('/advisor')
def history(project_id: str, request: Request, principal: Principal = Depends(require_principal)):
    return service(request).history(principal.account_id, project_id)


@router.post('/advisor/messages')
async def message(project_id: str, body: MessageRequest, request: Request, principal: Principal = Depends(require_principal)):
    provider = getattr(request.app.state, 'advisor_provider', None) or AdvisorProvider()
    return await service(request).ask(provider, request.app.state.settings, principal.account_id,
                                      project_id, body.message, body.request_key, body.job_id, body.dataset_id)


@router.post('/advisor/plans/{plan_id}/approve')
def approve(project_id: str, plan_id: str, request: Request, principal: Principal = Depends(require_principal)):
    return service(request).approve_plan(principal.account_id, project_id, plan_id)


@router.post('/advisor/evaluations')
def evaluation(project_id: str, body: EvaluationRequest, request: Request, principal: Principal = Depends(require_principal)):
    return service(request).evaluation(principal.account_id, project_id, body.job_id, body.criteria, body.observations)


@router.get('/datasets/{dataset_id}/inspect')
def inspect(project_id: str, dataset_id: str, request: Request, principal: Principal = Depends(require_principal)):
    return inspect_dataset(request.app.state.store, request.app.state.settings,
                           principal.account_id, project_id, dataset_id)
